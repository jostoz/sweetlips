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


from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from services.aec_filter import FarEndBuffer, WasapiLoopbackCapture, WebRTCAECFilter
from services.firered_vad import FireRedVADAnalyzer
from services.jev_system1 import JevSystem1Processor
from services.kokoro_gpu_tts import KokoroGPUTTSService
from services import latency_probe
from services.r2t2_stt import ConfuciusR2T2Service
from services.system2_llm import System2PromptBridge, System2ResponseCollector, build_shared_context


async def main():
    latency_probe.start_metrics_server(port=9091)

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
            # AEC (WebRTC AEC3) con referencia real por WASAPI loopback (ver
            # services/aec_filter.py) -- el primer intento (tapear frames
            # del pipeline de TTS) tenía un delay far-end impredecible y
            # degradaba el audio; el loopback captura lo que realmente
            # suena por el hardware, mismo dominio de tiempo que el mic.
            audio_in_filter=WebRTCAECFilter(far_end_buffer),
        )
    )

    # 2. Servicios.
    vad = VADProcessor(
        vad_analyzer=FireRedVADAnalyzer(
            params=VADParams(
                # 0.2s (default) corta la frase en pedacitos con cualquier
                # micro-pausa/respiración natural. Bajado a 0.5s en un
                # momento para ganar latencia, pero el patrón real en vivo
                # fue: se pierde sistemáticamente la palabra clave justo
                # después de una preposición ("historia de", "ciudad de",
                # "sabes algo de", siempre cortado ahí) -- el usuario hace
                # una micro-pausa pensando la palabra siguiente y el turno
                # se da por terminado antes de tiempo. Vuelta a 0.7s:
                # precisión > unos ms de latencia.
                stop_secs=0.7,
            )
        )
    )  # Capa 0: VAD acústico (FireRedVAD streaming).
    # Servidor R2T2 corriendo dentro de WSL2 (ver README/ws): vLLM no soporta
    # Windows nativo, por eso el motor vive en Linux y este cliente le habla
    # por WebSocket. Arrancar antes: wsl -e bash -lc "cd ~/Confucius4-R2T2 && ./run_start_server.sh start --model_path ~/models/Confucius4-R2T2"
    r2t2_stt = ConfuciusR2T2Service(
        # 127.0.0.1 explícito, no "localhost": en esta máquina Windows
        # "localhost" resuelve primero a IPv6 (::1), que no responde, y
        # requests/urllib3 tarda ~2s en caer a IPv4 antes de conectar.
        ws_uri="ws://127.0.0.1:8272/asr_stream_api_v1",
        language="Spanish",  # revertido: la prueba con "English" confirmó
        # que R2T2 transcribe frases MÁS COMPLETAS en inglés, pero rompe
        # el reconocimiento de los comandos de interrupción/acción, que
        # están en español ("cállate", "párate" llegaban irreconocibles,
        # ej. "Para. Hey", con el modelo primeado para inglés). Mientras
        # los comandos y el LLM sigan en español, R2T2 tiene que estar en
        # español -- el hallazgo del idioma queda documentado para si
        # algún día se hace una versión en inglés completa.
        # Hotwords (system_prompt) REVERTIDO: confirmado en vivo por el
        # usuario que causaba alucinaciones -- el modelo "escuchaba"
        # exactamente la lista de hotwords completa ("cállate, detente,
        # silencio, cancela, enciende la luz, apaga la luz") de forma
        # repetida e idéntica incluso cuando el usuario NO las había
        # dicho (confirmado: "yo no dije apaga la luz nunca"). El
        # contexto demasiado fuerte sesga al modelo a "oír" lo que se le
        # primea. No usar system_prompt con frases completas -- si se
        # reintenta, probar con palabras sueltas y bajo peso.
    )
    # Smart-turn: modelo ONNX (viene empaquetado con pipecat, sin
    # descarga) que decide semánticamente si el usuario terminó de
    # hablar, en vez de contar un timer fijo de silencio. Reemplaza el
    # trade-off "timer corto = rápido pero corta palabras" / "timer largo
    # = preciso pero siempre lento" por una decisión real por turno.
    smart_turn = LocalSmartTurnAnalyzerV3()
    jev_router = JevSystem1Processor(r2t2_stt=r2t2_stt, smart_turn=smart_turn)

    # Capa 3: System 2 (razonamiento). Groq (cloud, API compatible con
    # OpenAI, inferencia LPU muy rápida) para no competir por VRAM con el
    # servidor R2T2 en la GPU local.
    system2_context = build_shared_context()
    system2_prompt_bridge = System2PromptBridge(system2_context)
    system2_llm = OpenAILLMService(
        settings=OpenAILLMService.Settings(
            # qwen/qwen3.8-27b en vez de openai/gpt-oss-120b: medido 3x más
            # rápido (~220ms vs ~630ms) y SIN tokens de razonamiento -- oss-120b
            # es un modelo "reasoning" que se come el presupuesto de tokens
            # pensando (llegó a dejar respuestas de 1 letra, o vacías del
            # todo con oss-20b: 148/150 tokens gastados en razonamiento) y
            # tiene un bug conocido (reportado en vLLM/LangChain/HF) donde
            # el streaming del razonamiento rompe el parser de Groq y el
            # turno se pierde entero ("Parsing failed", visto en vivo en
            # esta sesión). qwen3.8-27b no es reasoning: no tiene ninguno
            # de los dos problemas.
            model="qwen/qwen3.8-27b",
            max_completion_tokens=55,
        ),
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    system2_response_collector = System2ResponseCollector(system2_context)

    # Kokoro-FastAPI (servidor separado, PyTorch+CUDA real -- medido:
    # síntesis en 135-230ms en GPU, vs ~1.2s en CPU vía onnxruntime).
    # Arrancar antes: vendor/Kokoro-FastAPI/start-gpu.ps1 (puerto 8880).
    tts = KokoroGPUTTSService(
        base_url="http://127.0.0.1:8880/v1",
        api_key="not-needed",
        model="kokoro",
        # af_heart era inglés (EEUU) -- sonaba como "americano hablando
        # español mal". ef_dora es una de las 3 voces en español que
        # trae Kokoro (ef_dora, em_alex, em_santa).
        voice="ef_dora",
    )
    loopback_capture = WasapiLoopbackCapture(far_end_buffer)
    await loopback_capture.start()  # referencia far-end real (WASAPI loopback) para el AEC.

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
            transport.output(),  # Altavoz físico.
        ]
    )

    task = PipelineTask(pipeline, enable_rtvi=False, idle_timeout_secs=None)
    runner = PipelineRunner()

    print("\n[Listo] El agente de voz Edge está escuchando... (Ctrl+C para salir)\n")
    try:
        await runner.run(task)
    finally:
        await loopback_capture.stop()


if __name__ == "__main__":
    asyncio.run(main())
