"""Cancelación de eco acústico (AEC) para poder usar parlantes en vez de
auriculares, como en un smart speaker (Alexa/Google Home).

Usa `pywebrtc-audio` (bindings de pybind11 sobre el módulo de audio de
WebRTC, el mismo AEC3 que usan navegadores y varios asistentes de voz
comerciales): https://pypi.org/project/pywebrtc-audio/

## Por qué el primer intento (tapear frames del pipeline) no funcionaba

La primera versión de este archivo armaba la señal de referencia
("far-end", lo que suena por el parlante) enganchándose a los
`OutputAudioRawFrame` que pasan por el pipeline de pipecat, justo antes
de `transport.output()`. El problema: esos frames se generan cuando el
TTS los produce, MUCHO antes de que realmente suenen por el hardware —
hay colas internas (buffer de reproducción de pipecat, buffer de
PyAudio, driver de audio) cuyo tamaño varía con la duración de la
respuesta. Con ese delay desconocido y variable, no hay forma de alinear
correctamente "lo que el mic capta ahora" con "lo que sonó hace X ms" —
el AEC3 diverge y degrada el audio en vez de limpiarlo (medido: rompía
transcripciones incluso con la señal de por sí limpia).

## Fix: WASAPI loopback (Windows) como referencia real

En vez de adivinar el delay del pipeline de TTS, `WasapiLoopbackCapture`
graba directamente lo que el sistema operativo está mandando al DAC
(`pyaudiowpatch`, un fork de PyAudio con soporte de loopback WASAPI) —
es decir, literalmente el mismo audio que sale por el parlante, captado
en el mismo dominio de tiempo (hardware) que el micrófono. El delay
remanente es sólo el de los buffers de audio (típicamente 10-40ms),
chico y estable turno a turno, no minutos de cola de TTS.

Arquitectura:
  - `WasapiLoopbackCapture`: stream de PyAudio en modo bloqueante sobre
    el dispositivo `[Loopback]` del output por defecto, corriendo en un
    executor (no bloquea el event loop). Cada chunk se downmixea a mono
    y se resamplea a 16kHz, y se escribe al buffer far-end compartido.
  - `WebRTCAECFilter`: se engancha en
    `LocalAudioTransportParams(audio_in_filter=...)`. Por cada chunk que
    llega del micrófono ("near-end"), saca del buffer far-end la misma
    cantidad de muestras y le resta el eco con
    `EchoCanceller.process(near, far)`.

Requiere Windows (WASAPI). `main.py` debe llamar
`await loopback.start()` al arrancar el pipeline y `await loopback.stop()`
al salir.
"""

from __future__ import annotations

import asyncio
import queue
import time

import numpy as np
from loguru import logger
from pywebrtc_audio import EchoCanceller

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

_AEC_SAMPLE_RATE = 16000
# Tope del buffer far-end. El delay físico real esperado (loopback +
# resampler + buffers de audio) es de 10-40ms -- 5s (valor original) era
# un error real: permitía que el backlog creciera sin control y se
# quedara ahí. Medido en vivo: far_buffer_pendiente se estabilizaba en
# ~4.4s CONSTANTES durante toda la sesión (WasapiLoopbackCapture escribe
# más rápido de lo que filter() lee), y el AEC terminaba comparando el
# mic contra audio que sonó hace 4+ segundos -- desalineación masiva.
#
# Bajarlo a 300ms (primer intento) resultó DEMASIADO agresivo: el
# loopback no entrega en flujo continuo estable, sino con huecos reales
# entre escrituras (instrumentado en _drain(), ver "loopback write" en
# los logs) -- con 300ms de margen el buffer quedaba casi vacío durante
# esos huecos, y el AEC terminaba comparando contra silencio (ERLE
# negativo: near_rms=2 -> cleaned_rms=12). 800ms es un punto medio
# mientras se mide la cadencia real (ver logs "[AEC diag] loopback
# write") para dimensionar esto con datos, no adivinando de nuevo.
_MAX_BUFFER_BYTES = _AEC_SAMPLE_RATE * 2 * 8 // 10  # 800ms


class FarEndBuffer:
    """FIFO de bytes PCM16 mono @ 16kHz compartido entre la captura de
    loopback y el filtro de AEC."""

    def __init__(self):
        self._buffer = bytearray()
        self._lock = asyncio.Lock()

    async def write(self, pcm16_bytes: bytes) -> None:
        async with self._lock:
            self._buffer.extend(pcm16_bytes)
            overflow = len(self._buffer) - _MAX_BUFFER_BYTES
            if overflow > 0:
                del self._buffer[:overflow]

    async def read(self, num_bytes: int) -> bytes:
        """Devuelve exactamente `num_bytes`; rellena con silencio si no hay
        suficiente far-end en el buffer (nada sonando por el parlante).

        Si el backlog acumulado supera el delay físico esperado (ver
        _MAX_BUFFER_BYTES), DESCARTA lo viejo antes de leer en vez de
        devolver la muestra más antigua -- eso es exactamente lo que
        causaba la desalineación de 4.4s: leer siempre desde el extremo
        viejo de un buffer que crece más rápido de lo que se consume."""
        async with self._lock:
            excess = len(self._buffer) - max(num_bytes, _MAX_BUFFER_BYTES)
            if excess > 0:
                del self._buffer[:excess]
            available = self._buffer[:num_bytes]
            del self._buffer[: len(available)]
        if len(available) < num_bytes:
            available = available + b"\x00" * (num_bytes - len(available))
        return bytes(available)

    async def pending_bytes(self) -> int:
        async with self._lock:
            return len(self._buffer)


class WasapiLoopbackCapture:
    """Graba en vivo lo que sale por el parlante por defecto (WASAPI
    loopback) y lo vuelca -downmixeado a mono, resampleado a 16kHz- al
    `FarEndBuffer` compartido con `WebRTCAECFilter`."""

    def __init__(self, far_end_buffer: FarEndBuffer, chunk_ms: int = 20):
        self._far_end_buffer = far_end_buffer
        self._chunk_ms = chunk_ms
        self._resampler = create_stream_resampler()
        self._pa = None
        self._stream = None
        self._device_rate = 0
        self._device_channels = 0
        self._queue: "queue.Queue[bytes | None]" = queue.Queue()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        speakers = self._pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        if not speakers["isLoopbackDevice"]:
            for loopback in self._pa.get_loopback_device_info_generator():
                if speakers["name"] in loopback["name"]:
                    speakers = loopback
                    break
            else:
                raise RuntimeError(
                    f"No se encontró el dispositivo de loopback WASAPI para "
                    f"'{speakers['name']}'. AEC requiere Windows con WASAPI."
                )

        self._device_rate = int(speakers["defaultSampleRate"])
        self._device_channels = int(speakers["maxInputChannels"])
        chunk_frames = int(self._device_rate * self._chunk_ms / 1000)

        def _callback(in_data, frame_count, time_info, status):
            # Corre en el hilo de PortAudio (C), no en el event loop: sólo
            # encolar, nada de asyncio acá. PortAudio invoca esto SÓLO
            # cuando el endpoint de render tiene audio activo -- si nada
            # suena por el parlante, simplemente no se llama (bloqueante
            # `stream.read()` se probó primero y colgaba para siempre en
            # silencio total: WASAPI no entrega paquetes de un endpoint
            # idle).
            self._queue.put_nowait(in_data)
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._device_channels,
            rate=self._device_rate,
            frames_per_buffer=chunk_frames,
            input=True,
            input_device_index=speakers["index"],
            stream_callback=_callback,
        )
        self._stream.start_stream()
        logger.info(
            f"[AEC] Loopback WASAPI sobre '{speakers['name']}' "
            f"@ {self._device_rate}Hz x{self._device_channels}ch -> "
            f"far-end 16kHz mono."
        )
        self._task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            while True:
                data = await loop.run_in_executor(None, self._queue.get)
                if data is None:
                    # Sentinel de stop(): salir limpio (ver comentario ahí
                    # sobre por qué no alcanza con cancel() solo).
                    return
                # Auditoría (misma línea que el bug del FarEndBuffer, ver
                # README/commit "causa raiz real"): esta cola cruda entre
                # el callback de PortAudio y el resampler nunca tuvo tope
                # ni instrumentación -- si _drain() se atrasa (resampler,
                # congestión del event loop), acumula acá ANTES de llegar
                # al FarEndBuffer (que ya recorta agresivo). Impacto ya
                # mitigado río abajo, pero se loguea si crece para no
                # repetir el mismo error de "asumir sin medir".
                backlog = self._queue.qsize()
                if backlog > 10:
                    logger.warning(
                        f"[AEC] Cola de loopback atrasada: {backlog} chunks pendientes "
                        f"(~{backlog * self._chunk_ms}ms) -- el resampler o el event loop "
                        f"no están siguiendo el ritmo del audio real."
                    )
                mono = self._downmix(data)
                pcm16k = await self._resampler.resample(
                    mono, self._device_rate, _AEC_SAMPLE_RATE
                )
                # Diagnóstico: mismo error que el buffer 5s original --
                # no volver a adivinar un tope sin medir la cadencia real
                # de escritura. Loguea cada burst con cuánto pasó desde
                # el anterior y cuántos bytes trajo -- si el loopback
                # entrega en ráfagas espaciadas (no un flujo continuo
                # ~1x tiempo real), el tope del buffer tiene que
                # dimensionarse para el HUECO entre ráfagas, no para el
                # "jitter" que se asumía al principio.
                now = time.monotonic()
                last = getattr(self, "_last_write_time", None)
                gap_ms = (now - last) * 1000 if last is not None else 0.0
                self._last_write_time = now
                if gap_ms > 100:
                    logger.debug(
                        f"[AEC diag] loopback write: {len(pcm16k)}B tras "
                        f"{gap_ms:.0f}ms de hueco desde la escritura anterior"
                    )
                await self._far_end_buffer.write(pcm16k)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[AEC] Loopback capture error: {e}")

    def _downmix(self, data: bytes) -> bytes:
        if self._device_channels == 1:
            return data
        arr = np.frombuffer(data, dtype=np.int16).reshape(-1, self._device_channels)
        return arr.mean(axis=1).astype(np.int16).tobytes()

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
        if self._task is not None:
            # `queue.get()` bloqueante corriendo en el executor no se
            # interrumpe con task.cancel() hasta que retorna -- mandamos
            # un sentinel para desbloquearlo, después cancel() como red
            # de seguridad por si ya estaba en otro punto.
            self._queue.put_nowait(None)
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None




class WebRTCAECFilter(BaseAudioFilter):
    """Filtro de entrada: resta el eco del parlante usando AEC3 (WebRTC)."""

    def __init__(self, far_end_buffer: FarEndBuffer, stream_delay_ms: int = 0):
        self._far_end_buffer = far_end_buffer
        # 0 = dejar que el estimador de delay interno de AEC3 lo calcule.
        # Con loopback real el delay es chico y estable (buffers de audio,
        # no la cola de TTS), así que el estimador converge rápido sin
        # necesidad de adivinar un valor fijo.
        self._stream_delay_ms = stream_delay_ms
        self._aec: EchoCanceller | None = None
        self._sample_rate = 0
        self._enabled = True
        self._far_was_silent = True
        """Diagnóstico en vivo (ver README, sección AEC): cancela bien en
        volumen bajo/moderado (hasta ~90% de reducción de RMS) pero en
        picos fuertes deja de cancelar o directamente AMPLIFICA
        (near_rms=302 -> cleaned_rms=590) -- firma clásica de un filtro
        adaptativo con el delay desalineado, sumando en vez de restar.
        El WASAPI loopback callback solo dispara cuando hay audio activo
        (ver WasapiLoopbackCapture): cada transición silencio->sonido es
        un punto de discontinuidad donde el estado adaptativo de AEC3,
        convergido para el turno anterior, puede quedar desalineado para
        el nuevo. Resetear ahí fuerza una reconvergencia limpia en vez de
        arrastrar un estado potencialmente stale/incorrecto."""

    async def start(self, sample_rate: int) -> None:
        if sample_rate != _AEC_SAMPLE_RATE:
            raise ValueError(
                f"WebRTCAECFilter requiere {_AEC_SAMPLE_RATE} Hz de entrada "
                f"(recibido: {sample_rate}). Ajustar audio_in_sample_rate."
            )
        self._sample_rate = sample_rate
        self._aec = EchoCanceller(
            sample_rate=sample_rate, num_channels=1, stream_delay_ms=self._stream_delay_ms
        )
        logger.debug(f"[AEC] EchoCanceller iniciado @ {sample_rate}Hz, delay={self._stream_delay_ms}ms")

    async def stop(self) -> None:
        self._aec = None

    def reset(self) -> None:
        """Fuerza al filtro adaptativo a reconverger desde cero. Llamado
        automáticamente en cada transición silencio->sonido del far-end
        (ver filter()); expuesto también para que main.py lo dispare
        explícitamente en BotStartedSpeakingFrame como red adicional."""
        if self._aec is not None:
            self._aec.reset()

    async def process_frame(self, frame: FilterControlFrame) -> None:
        if isinstance(frame, FilterEnableFrame):
            self._enabled = frame.enable

    _DIAG_LOG_EVERY = 50
    """Cada cuántos chunks (con far-end activo) loguear métricas de
    diagnóstico -- a 16kHz/20ms por chunk, ~1s. Sin este throttle,
    loguear cada chunk inunda el log (50/s)."""

    async def filter(self, audio: bytes) -> bytes:
        if not self._enabled or self._aec is None:
            return audio

        far = await self._far_end_buffer.read(len(audio))
        far_arr = np.frombuffer(far, dtype=np.int16)

        if not np.any(far_arr):
            # No hay nada sonando por el parlante ahora mismo: sin eco que
            # cancelar. Procesar igual degradaba el audio limpio (medido:
            # -25% RMS en silencio de far-end), así que devolvemos el audio
            # del mic sin tocar.
            self._far_was_silent = True
            return audio

        if self._far_was_silent:
            # Transición silencio -> sonido: ver docstring de __init__.
            # Reconvergencia limpia en vez de arrastrar el estado
            # adaptativo (posiblemente desalineado) del turno anterior.
            self.reset()
            self._far_was_silent = False

        near_arr = np.frombuffer(audio, dtype=np.int16)

        try:
            cleaned = self._aec.process(near_arr, far_arr)
            cleaned_arr = np.asarray(cleaned, dtype=np.int16)

            # Diagnóstico: ¿cuánto está cancelando realmente? ERLE
            # (Echo Return Loss Enhancement) aproximado: cuánto bajó el
            # RMS del near-end después de restar el eco estimado. Un AEC
            # que cancela bien debería mostrar ERLE de varios dB cuando
            # hay far-end fuerte; ERLE ~0 con far-end fuerte = está
            # dejando pasar el eco casi intacto.
            self._diag_counter = getattr(self, "_diag_counter", 0) + 1
            if self._diag_counter % self._DIAG_LOG_EVERY == 0:
                near_rms = float(np.sqrt(np.mean(near_arr.astype(np.float64) ** 2)) + 1e-6)
                cleaned_rms = float(np.sqrt(np.mean(cleaned_arr.astype(np.float64) ** 2)) + 1e-6)
                far_rms = float(np.sqrt(np.mean(far_arr.astype(np.float64) ** 2)) + 1e-6)
                erle_db = 20 * np.log10(near_rms / cleaned_rms) if cleaned_rms > 0 else 0.0
                pending = await self._far_end_buffer.pending_bytes()
                logger.debug(
                    f"[AEC diag] near_rms={near_rms:.0f} cleaned_rms={cleaned_rms:.0f} "
                    f"far_rms={far_rms:.0f} erle={erle_db:+.1f}dB "
                    f"far_buffer_pendiente={pending}B ({pending / (_AEC_SAMPLE_RATE * 2) * 1000:.0f}ms)"
                )

            return cleaned_arr.tobytes()
        except Exception as e:
            logger.error(f"[AEC] Error procesando audio: {e}")
            return audio
