"""Input transport de micrófono: abre WASAPI a su tasa NATIVA (48kHz) y
resamplea a la tasa que pide el pipeline (16kHz) en el propio proceso.

Escrito originalmente para que EchoNull (AEC por GPU vía Equalizer APO)
pudiera interceptar el mic -- ESA parte se descartó (EchoNull cancelaba
la voz real del usuario, ver README "Migración AEC"), pero esta clase
se sigue usando igual, por una razón distinta encontrada después:
**instalar Equalizer APO rompió el backend MME** para este dispositivo
(medido con captura cruda, tools/wasapi_level_check.py: pico 35-44/32767
hablando fuerte, a cualquier tasa -- 16kHz forzado o nativa 44100Hz,
mismo resultado). WASAPI captura bien (pico 32502/32767, mismo hardware,
mismo momento) -- es la única vía de entrada utilizable ahora, aunque no
se use ningún AEC de terceros.

`audio_in_filter` (nuestro WebRTCAECFilter, ver aec_filter.py) SÍ se
sigue aplicando normal con este transport: usa `push_audio_frame()`,
que alimenta la misma cola genérica de `BaseInputTransport` donde
pipecat aplica el filtro (confirmado leyendo
pipecat/transports/base_input.py) -- no hace falta reimplementar nada
de eso acá.
"""

from __future__ import annotations

import asyncio

import pyaudio
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.transports.local.audio import LocalAudioInputTransport, LocalAudioTransportParams


class WASAPIResampledInputTransport(LocalAudioInputTransport):
    """Como LocalAudioInputTransport, pero abre el stream a `native_rate`
    (la tasa real del dispositivo WASAPI) y resamplea cada chunk a la tasa
    que el resto del pipeline espera (`params.audio_in_sample_rate`) antes
    de armar el `InputAudioRawFrame` -- así el frame reporta el sample_rate
    correcto Y el contenido de audio corresponde de verdad a esa tasa (un
    `audio_in_filter` no puede lograr esto: solo toca `frame.audio`,
    `frame.sample_rate` ya quedó fijado antes de que el filtro corra)."""

    def __init__(
        self, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams, native_rate: int
    ):
        super().__init__(py_audio, params)
        self._native_rate = native_rate
        self._resampler = create_stream_resampler()

    async def setup(self, setup: FrameProcessorSetup):
        # Salteamos LocalAudioInputTransport.setup() (abre el stream con
        # self._sample_rate == la tasa de SALIDA, no la nativa) y
        # replicamos su lógica con native_rate para el open() real.
        await super(LocalAudioInputTransport, self).setup(setup)

        num_frames = int(self._native_rate / 100) * 2  # 20ms de audio, a la tasa nativa.

        self._in_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_in_channels,
            rate=self._native_rate,
            frames_per_buffer=num_frames,
            stream_callback=self._audio_in_callback,
            input=True,
            input_device_index=self._params.input_device_index,
        )
        logger.info(
            f"[WASAPIResampledInput] Mic abierto @ {self._native_rate}Hz nativo, "
            f"resampleando a {self._sample_rate}Hz para el pipeline."
        )

    def _audio_in_callback(self, in_data, frame_count, time_info, status):
        # Corre en el hilo de PortAudio (C) -- el resampler es async, así
        # que lo despachamos al loop en vez de bloquear este callback.
        asyncio.run_coroutine_threadsafe(self._resample_and_push(in_data), self.get_event_loop())
        return (None, pyaudio.paContinue)

    async def _resample_and_push(self, raw_audio: bytes) -> None:
        resampled = await self._resampler.resample(raw_audio, self._native_rate, self._sample_rate)
        if not resampled:
            return
        frame = InputAudioRawFrame(
            audio=resampled,
            sample_rate=self._sample_rate,
            num_channels=self._params.audio_in_channels,
        )
        await self.push_audio_frame(frame)

