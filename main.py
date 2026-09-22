"""Pipeline Edge: micrófono local -> FireRedVAD -> R2T2 (ASR append-only)
-> Jev (System 1) -> acción local / LLM cloud (System 2, Groq) ->
-> Kokoro TTS local -> altavoz.

Requiere pipecat-ai>=1.9.0 (API de servicios/transportes actual) y el
paquete `fireredvad` (no está en PyPI; ver services/firered_vad.py). Ver
README de cada servicio para instalar sus extras:
    pip install -r requirements.txt

El LLM de System 2 ya NO es local: apunta a Groq (API compatible con
OpenAI, inferencia LPU de baja latencia). Requiere la variable de entorno
GROQ_API_KEY con una key válida de https://console.groq.com/keys. Se sacó
LM Studio del pipeline porque competía por VRAM con el servidor R2T2
(vLLM) en la misma GPU.
"""

import asyncio
import os


from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from services.aec_filter import FarEndBuffer, FarEndTapProcessor, WebRTCAECFilter
from services.firered_vad import FireRedVADAnalyzer
from services.jev_system1 import JevSystem1Processor
from services.r2t2_stt import ConfuciusR2T2Service
from services.system2_llm import System2PromptBridge, System2ResponseCollector, build_shared_context


async def main():
    # 1. Audio local (micro y altavoz físicos del equipo).
    far_end_buffer = FarEndBuffer()  # señal de referencia para el AEC.
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # Usa el micrófono default de Windows (Micrófono Steren COM-126,
            # verificado funcional). El Realtek USB Audio aparecía
            # desconectado ("Unknown" en Device Manager) al probarlo.
            audio_in_sample_rate=16000,
            # AEC (WebRTC AEC3): cancela el eco del propio parlante para
            # poder usar altavoces en vez de auriculares, como un smart
            # speaker. Necesita la señal de referencia (far_end_tap más
            # abajo, antes de transport.output()).
            audio_in_filter=WebRTCAECFilter(far_end_buffer),
        )
    )

    # 2. Servicios.
    vad = VADProcessor(
        vad_analyzer=FireRedVADAnalyzer(
            params=VADParams(
                # 0.2s (default) corta la frase en pedacitos con cualquier
                # micro-pausa/respiración natural. 0.7s tolera pausas reales
                # sin cortar el turno a mitad de oración.
                stop_secs=0.7,
            )
        )
    )  # Capa 0: VAD acústico (FireRedVAD streaming).
    # Servidor R2T2 corriendo dentro de WSL2 (ver README/ws): vLLM no soporta
    # Windows nativo, por eso el motor vive en Linux y este cliente le habla
    # por WebSocket. Arrancar antes: wsl -e bash -lc "cd ~/Confucius4-R2T2 && ./run_start_server.sh start --model_path ~/models/Confucius4-R2T2"
    r2t2_stt = ConfuciusR2T2Service(
        ws_uri="ws://localhost:8272/asr_stream_api_v1",
        language="Spanish",  # forzado: evita el modo bilingüe zh/en por defecto.
    )
    jev_router = JevSystem1Processor()

    # Capa 3: System 2 (razonamiento). Groq (cloud, API compatible con
    # OpenAI, inferencia LPU muy rápida) para no competir por VRAM con el
    # servidor R2T2 en la GPU local.
    system2_context = build_shared_context()
    system2_prompt_bridge = System2PromptBridge(system2_context)
    system2_llm = OpenAILLMService(
        settings=OpenAILLMService.Settings(model="openai/gpt-oss-120b"),
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    system2_response_collector = System2ResponseCollector(system2_context)

    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice="af_heart", language=Language.ES),
    )
    far_end_tap = FarEndTapProcessor(far_end_buffer)  # alimenta la señal de referencia del AEC.

    # 3. Pipeline.
    pipeline = Pipeline(
        [
            transport.input(),  # Micro local (16 kHz) + AEC.
            vad,  # Capa 0: VAD acústico -> VADUserStarted/StoppedSpeakingFrame.
            r2t2_stt,  # Capa 1: ASR streaming append-only.
            jev_router,  # Capa 2: System 1 (decisión/interrupción/filtro).
            system2_prompt_bridge,  # Capa 3a: arma el turno de LLM cuando Jev escala.
            system2_llm,  # Capa 3b: LLM cloud (Groq, OpenAI-compatible).
            system2_response_collector,  # Capa 3c: guarda la respuesta en el contexto.
            tts,  # Capa 4: streaming TTS.
            far_end_tap,  # Captura lo que va a sonar, para el AEC.
            transport.output(),  # Altavoz físico.
        ]
    )

    task = PipelineTask(pipeline, enable_rtvi=False)
    runner = PipelineRunner()

    print("\n[Listo] El agente de voz Edge está escuchando... (Ctrl+C para salir)\n")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
