"""Cancelación de eco acústico (AEC) para poder usar parlantes en vez de
auriculares, como en un smart speaker (Alexa/Google Home).

Usa `pywebrtc-audio` (bindings de pybind11 sobre el módulo de audio de
WebRTC, el mismo AEC3 que usan navegadores y varios asistentes de voz
comerciales): https://pypi.org/project/pywebrtc-audio/

Arquitectura (dos piezas que se pasan un buffer compartido):
  - `FarEndTapProcessor`: se coloca justo antes de `transport.output()`.
    Cada `OutputAudioRawFrame`/`TTSAudioRawFrame` que va a sonar por el
    parlante se resamplea a 16 kHz y se encola en el buffer "far-end"
    (la señal de referencia: lo que el sistema está reproduciendo).
  - `WebRTCAECFilter`: se engancha en
    `LocalAudioTransportParams(audio_in_filter=...)`. Por cada chunk que
    llega del micrófono ("near-end"), saca del buffer far-end la misma
    cantidad de muestras (FIFO — el orden de encolado ya refleja el orden
    de reproducción) y le resta el eco con `EchoCanceller.process(near, far)`.

Nota de calibración: `stream_delay_ms` estima cuánto tarda el audio en ir
desde que se genera hasta que el micrófono lo capta de vuelta (buffers de
PyAudio + latencia del hardware). El valor por defecto (150 ms) es un punto
de partida razonable para audio USB genérico, pero puede necesitar ajuste
empírico: si el eco no se cancela bien, probar subir/bajar este valor.
"""

from __future__ import annotations

import asyncio

import numpy as np
from loguru import logger
from pywebrtc_audio import EchoCanceller

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    FilterControlFrame,
    FilterEnableFrame,
    Frame,
    OutputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_AEC_SAMPLE_RATE = 16000
# Tope del buffer far-end: ~5s de audio a 16kHz/16bit mono. Suficiente para
# absorber jitter sin crecer indefinidamente si el filtro deja de consumir.
_MAX_BUFFER_BYTES = _AEC_SAMPLE_RATE * 2 * 5


class FarEndBuffer:
    """FIFO de bytes PCM16 mono @ 16kHz compartido entre el tap y el filtro."""

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
        suficiente far-end en el buffer (el bot no está hablando)."""
        async with self._lock:
            available = self._buffer[:num_bytes]
            del self._buffer[: len(available)]
        if len(available) < num_bytes:
            available = available + b"\x00" * (num_bytes - len(available))
        return bytes(available)

    async def pending_bytes(self) -> int:
        async with self._lock:
            return len(self._buffer)


class FarEndTapProcessor(FrameProcessor):
    """Captura el audio de salida (TTS) y lo vuelca al buffer far-end."""

    def __init__(self, far_end_buffer: FarEndBuffer, **kwargs):
        super().__init__(**kwargs)
        self._far_end_buffer = far_end_buffer
        self._resampler = create_stream_resampler()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputAudioRawFrame) and frame.audio:
            pcm16k = await self._resampler.resample(
                frame.audio, frame.sample_rate, _AEC_SAMPLE_RATE
            )
            await self._far_end_buffer.write(pcm16k)

        await self.push_frame(frame, direction)


class WebRTCAECFilter(BaseAudioFilter):
    """Filtro de entrada: resta el eco del parlante usando AEC3 (WebRTC)."""

    def __init__(self, far_end_buffer: FarEndBuffer, stream_delay_ms: int = 150):
        self._far_end_buffer = far_end_buffer
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
