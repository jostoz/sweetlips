"""TTS nativo de Windows vía SAPI5 (win32com), incluyendo las voces
"Natural" (Ava, Jenny, Dalia, Jorge...) desbloqueadas por
NaturalVoiceSAPIAdapter (https://github.com/gexgd0419/NaturalVoiceSAPIAdapter,
MIT, instalado fuera de este repo en esta máquina -- ver README, sección
"TTS nativo de Windows").

Motivación (sesión del 2026-09-23): investigamos reemplazar Kokoro-FastAPI
por algo sin GPU, para eliminar de raíz la contención de cómputo entre
Kokoro y R2T2 (ver README, "Contención residual de CÓMPUTO"). Las voces
SAPI5 nativas de Windows (OneCore clásicas: David/Zira/Mark/Sabina/Raul)
son gratis, locales, sin GPU, 30-61ms de latencia -- pero suenan a Windows
7/8. Las voces "Natural" de Windows 11 (Ava, Jenny, Aria...) suenan mucho
mejor, pero Microsoft las bloquea a propósito para que solo las use
Narrator (confirmado leyendo el AppxManifest.xml del paquete: están
registradas como `windows.appExtension` tipo `com.microsoft.voice.model.1`,
no como voz SAPI5/WinRT de propósito general). NaturalVoiceSAPIAdapter usa
claves de cifrado extraídas de archivos del sistema para desbloquearlas
igual -- es un hack no soportado por Microsoft, puede dejar de funcionar
en cualquier actualización de Windows. Usuario aceptó el riesgo
explícitamente tras ser advertido dos veces.

Dos familias de voces "Natural" quedan disponibles tras instalar el
adapter:
  - Locales (ej. "Microsoft Jenny (Natural)"): rápidas (~150ms), sin red,
    pero hay que descargar el archivo MSIX de la voz por separado (ver
    wiki del adapter) -- no todos los idiomas tienen versión local.
  - "Online" (ej. "Microsoft Ava Online (Natural)", "Microsoft Dalia
    Online (Natural)"): llamada de red al backend de "Leer en voz alta"
    de Edge, gratis, sin API key -- pero 1.2-1.9s de latencia medida en
    esta máquina, MÁS LENTO que Kokoro (170-300ms). Usuario eligió
    aceptar esa latencia a cambio de la calidad de voz.

La API de SAPI5 (`SAPI.SpVoice`) es SÍNCRONA/bloqueante y no soporta
streaming incremental real como la respuesta HTTP de Kokoro -- toda la
síntesis debe completarse antes de que lleguen los primeros bytes de
audio (`SpMemoryStream` los acumula en memoria). Por eso `run_tts` corre
la síntesis completa en un hilo aparte (`run_in_executor`, no bloquea el
event loop) y recién después trocea el resultado en frames -- el
time-to-first-audio real es la latencia total de síntesis, no menor.

COM (win32com) requiere `CoInitialize()`/`CoUninitialize()` por hilo --
los hilos del executor por defecto de asyncio son reutilizables, así que
se llama en cada invocación (CoInitialize repetido en el mismo hilo ya
inicializado es no-op seguro, documentado por Microsoft).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pythoncom
import win32com.client

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService, TTSSettings
from pipecat.utils.tracing.service_decorators import traced_tts

_SAMPLE_RATE = 16000
_SAFT_16kHz16BitMono = 18
# Enum SpeechAudioFormatType de SAPI5 (documentado por Microsoft):
# 18 = SAFT16kHz16BitMono. Verificado en vivo: bytes generados / 2 /
# 16000 coincide con la duración real del audio sintetizado.

_VOICE_NAME_BY_LANG = {
    "English": "Microsoft Ava Online (Natural)",
    "Spanish": "Microsoft Dalia Online (Natural)",
}


class WindowsTTSService(TTSService):
    """Cliente SAPI5 (win32com) para las voces nativas/Natural de Windows.

    No usa `pyttsx3` ni `System.Speech` (.NET): ambos tuvieron problemas
    reales en esta sesión (`SelectVoice` no encontraba las voces Natural
    por nombre exacto, aun estando listadas). `SAPI.SpVoice` (COM directo)
    sí funciona de forma confiable."""

    Settings = TTSSettings
    _settings: Settings

    def __init__(self, *, language: str = "English", voice: str | None = None, **kwargs):
        resolved_voice = voice or _VOICE_NAME_BY_LANG.get(language, _VOICE_NAME_BY_LANG["English"])
        super().__init__(
            sample_rate=_SAMPLE_RATE,
            settings=self.Settings(model=None, voice=resolved_voice, language=None),
            **kwargs,
        )
        self._voice_name = resolved_voice

    def can_generate_metrics(self) -> bool:
        return True

    def _synthesize_sync(self, text: str) -> bytes:
        """Bloqueante: corre en un hilo aparte via run_in_executor."""
        pythoncom.CoInitialize()
        try:
            sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
            target_token = None
            for token in sapi_voice.GetVoices():
                if self._voice_name in token.GetDescription():
                    target_token = token
                    break
            if target_token is None:
                raise RuntimeError(
                    f"Voz SAPI5 no encontrada: {self._voice_name!r} -- "
                    "¿está instalado NaturalVoiceSAPIAdapter? (ver README)"
                )
            sapi_voice.Voice = target_token

            mem_stream = win32com.client.Dispatch("SAPI.SpMemoryStream")
            audio_format = win32com.client.Dispatch("SAPI.SpAudioFormat")
            audio_format.Type = _SAFT_16kHz16BitMono
            mem_stream.Format = audio_format
            sapi_voice.AudioOutputStream = mem_stream

            sapi_voice.Speak(text)
            return bytes(mem_stream.GetData())
        finally:
            pythoncom.CoUninitialize()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_tts_usage_metrics(text)
            audio = await asyncio.get_event_loop().run_in_executor(None, self._synthesize_sync, text)
            await self.stop_ttfb_metrics()
            if not audio:
                return
            for i in range(0, len(audio), self.chunk_size):
                chunk = audio[i : i + self.chunk_size]
                if chunk:
                    yield TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
        except Exception as e:
            yield ErrorFrame(error=f"Windows TTS (SAPI5) error: {e}")
