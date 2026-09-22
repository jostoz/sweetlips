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

import numpy as np
from loguru import logger
from pywebrtc_audio import EchoCanceller

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

_AEC_SAMPLE_RATE = 16000
# Tope del buffer far-end: ~5s de audio a 16kHz/16bit mono. Suficiente para
# absorber jitter sin crecer indefinidamente si el filtro deja de consumir.
_MAX_BUFFER_BYTES = _AEC_SAMPLE_RATE * 2 * 5


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
        suficiente far-end en el buffer (nada sonando por el parlante)."""
        async with self._lock:
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
                mono = self._downmix(data)
                pcm16k = await self._resampler.resample(
                    mono, self._device_rate, _AEC_SAMPLE_RATE
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

    def _downmix(self, data: bytes) -> bytes:
        if self._device_channels == 1:
            return data
        arr = np.frombuffer(data, dtype=np.int16).reshape(-1, self._device_channels)
        return arr.mean(axis=1).astype(np.int16).tobytes()


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

    async def process_frame(self, frame: FilterControlFrame) -> None:
        if isinstance(frame, FilterEnableFrame):
            self._enabled = frame.enable

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
            return audio

        near_arr = np.frombuffer(audio, dtype=np.int16)

        try:
            cleaned = self._aec.process(near_arr, far_arr)
            return np.asarray(cleaned, dtype=np.int16).tobytes()
        except Exception as e:
            logger.error(f"[AEC] Error procesando audio: {e}")
            return audio
