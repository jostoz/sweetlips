"""TTS local por GPU: Kokoro-FastAPI (PyTorch+CUDA) vía su API
OpenAI-compatible (https://github.com/remsky/Kokoro-FastAPI).

`pipecat.services.openai.tts.OpenAITTSService` no sirve directo: valida el
nombre de voz contra una whitelist fija de voces de OpenAI (alloy, echo,
nova, ...) y rechaza nombres de Kokoro como "af_heart". Este wrapper es
básicamente el mismo `run_tts` sin esa validación, usando el cliente
`openai` (async, streaming) igual que la clase original.

Medido en esta máquina: síntesis real en GPU ~135-230ms (vs ~1.2s en CPU
vía onnxruntime). El cuello de botella anterior (2.2s totales) era
Windows resolviendo "localhost" a IPv6 primero — hay que usar 127.0.0.1.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
from loguru import logger
from openai import AsyncOpenAI, BadRequestError, DefaultAsyncHttpxClient

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService, TTSSettings
from pipecat.utils.tracing.service_decorators import traced_tts

_SAMPLE_RATE = 24000  # Kokoro nativo.


class KokoroGPUTTSService(TTSService):
    """Cliente para un servidor Kokoro-FastAPI local (GPU) vía su API
    OpenAI-compatible, sin la validación de voces de `OpenAITTSService`."""

    Settings = TTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8880/v1",
        api_key: str = "not-needed",
        model: str = "kokoro",
        voice: str = "af_heart",
        http_client: httpx.AsyncClient | None = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=_SAMPLE_RATE,
            settings=self.Settings(model=model, voice=voice, language=None),
            **kwargs,
        )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client or DefaultAsyncHttpxClient(),
        )

    def can_generate_metrics(self) -> bool:
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            async with self._client.audio.speech.with_streaming_response.create(
                input=text,
                model=self._settings.model,
                voice=self._settings.voice,
                response_format="pcm",
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(f"{self} error getting audio (status: {r.status_code}, error: {error})")
                    yield ErrorFrame(error=f"Error getting audio (status: {r.status_code}, error: {error})")
                    return

                await self.start_tts_usage_metrics(text)

                async for chunk in r.iter_bytes(self.chunk_size):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
        except BadRequestError as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
