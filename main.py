"""Pipeline Edge: micrófono local -> FireRedVAD -> R2T2 (ASR append-only)
-> Jev (System 1) -> acción local / LLM -> Kokoro TTS local -> altavoz.

Requiere pipecat-ai>=1.9.0 (API de servicios/transportes actual) y el
paquete `fireredvad` (no está en PyPI; ver services/firered_vad.py). Ver
README de cada servicio para instalar sus extras:
    pip install -r requirements.txt
"""

import asyncio

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from services.firered_vad import FireRedVADAnalyzer
from services.jev_system1 import JevSystem1Processor
from services.r2t2_stt import ConfuciusR2T2Service


async def main():
    # 1. Audio local (micro y altavoz físicos del equipo).
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # 2. Servicios.
    vad = VADProcessor(FireRedVADAnalyzer())  # Capa 0: VAD acústico (FireRedVAD streaming).
    # Servidor R2T2 corriendo dentro de WSL2 (ver README/ws): vLLM no soporta
    # Windows nativo, por eso el motor vive en Linux y este cliente le habla
    # por WebSocket. Arrancar antes: wsl -e bash -lc "cd ~/Confucius4-R2T2 && ./run_start_server.sh start --model_path ~/models/Confucius4-R2T2"
    r2t2_stt = ConfuciusR2T2Service(ws_uri="ws://localhost:8272/asr_stream_api_v1")
    jev_router = JevSystem1Processor()
    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice="af_heart", language=Language.ES),
    )

    # 3. Pipeline.
    pipeline = Pipeline(
        [
            transport.input(),  # Micro local (16 kHz).
            vad,  # Capa 0: VAD acústico -> VADUserStarted/StoppedSpeakingFrame.
            r2t2_stt,  # Capa 1: ASR streaming append-only.
            jev_router,  # Capa 2: System 1 (decisión/interrupción/filtro).
            # Si jev_router escala a System 2, aquí se intercalaría el LLMService.
            tts,  # Capa 4: streaming TTS.
            transport.output(),  # Altavoz físico.
        ]
    )

    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    print("\n[Listo] El agente de voz Edge está escuchando... (Ctrl+C para salir)\n")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
