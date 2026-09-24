"""Pipeline Edge: micrófono local -> FireRedVAD -> Nemotron 3.5 ASR
(streaming cache-aware) -> Jev (System 1) -> acción local / LLM cloud
(System 2, Groq) -> Windows TTS local -> altavoz.

Requiere pipecat-ai>=1.9.0 (API de servicios/transportes actual), el
paquete `fireredvad` (no está en PyPI; ver services/firered_vad.py) y
transformers>=5.13.0 (soporte de Nemotron3_5Asr). Ver README de cada
servicio para instalar sus extras:
    pip install -r requirements.txt

El LLM de System 2 ya NO es local: apunta a Groq (API compatible con
OpenAI, inferencia LPU de baja latencia). Requiere la variable de entorno
GROQ_API_KEY con una key válida de https://console.groq.com/keys.

Todo el pipeline corre nativo en Windows: el ASR anterior (R2T2) obligaba
a un servidor aparte dentro de WSL2 porque vLLM no soporta Windows --
Nemotron corre en-proceso vía Transformers y elimina esa dependencia.
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
from services.windows_tts import WindowsTTSService
from services import latency_probe
from services.nemotron_stt import NemotronASRService
from services.system2_llm import (
    DEFAULT_SYSTEM_PROMPT,
    System2PromptBridge,
    System2ResponseCollector,
    build_shared_context,
)


async def main():
    latency_probe.start_metrics_server(port=9091)

    # 1. Audio local (micro y altavoz físicos del equipo).
    far_end_buffer = FarEndBuffer()  # señal de referencia para el AEC.
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # Fijado explícito (no "default de Windows"): el equipo tiene
            # DOS micrófonos -- "Steren COM-126" (integrado a la webcam,
            # confirmado por su entrada duplicada como Camera en el
            # registro de dispositivos) y "Realtek USB Audio" (el externo
            # real, USB aparte). El comentario viejo acá decía que se
            # usaba Steren porque en su momento Realtek aparecía
            # desconectado en Device Manager -- eso cambió, Realtek ya
            # funciona y es el mic correcto a usar, pero depender del
            # "default de Windows" es frágil (cambia solo si se
            # conecta/desconecta algo, sin aviso). Índice 17 = "Micrófono
            # (Realtek USB Audio)" vía host API WASAPI (no MME/DirectSound/
            # WDM-KS, que pyaudio también expone como entradas separadas
            # para el mismo dispositivo físico). Probado índice 17
            # (WASAPI) primero por consistencia con el loopback del AEC,
            # pero WASAPI exclusive/shared no acepta 16kHz directo del
            # dispositivo (nativo 48kHz) -- "[Errno -9997] Invalid sample
            # rate", falla real en vivo. MME (índice 1) sí resamplea
            # automáticamente vía portaudio, que es lo que ya funcionaba
            # con el "default de Windows" anterior.
            # Verificar con
            # `python -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name'], p.get_host_api_info_by_index(p.get_device_info_by_index(i)['hostApi'])['name']) for i in range(p.get_device_count())]"`
            # si cambia el hardware.
            input_device_index=1,
            audio_in_sample_rate=16000,
            # AEC (WebRTC AEC3) con referencia real por WASAPI loopback (ver
            # services/aec_filter.py) -- el primer intento (tapear frames
            # del pipeline de TTS) tenía un delay far-end impredecible y
            # degradaba el audio; el loopback captura lo que realmente
            # suena por el hardware, mismo dominio de tiempo que el mic.
            #
            # stream_delay_ms medido con tools/calibrate_aec_delay.py
            # (chirp conocido, correlación cruzada contra el mic real --
            # mismo principio que un micrófono de calibración acústica
            # tipo Audyssey): 171.6/173.3/172.4ms en 3 corridas, SNR de
            # correlación hasta 270000x (medición muy confiable). Dejarlo
            # en 0 (estimador interno de AEC3 adivinando) coincidía con
            # los picos de ERLE negativo medidos hoy -- el delay real es
            # ~4x más grande que los "10-40ms" que se asumía antes sin
            # medir. Si cambia el hardware de audio, recalibrar.
            audio_in_filter=WebRTCAECFilter(far_end_buffer, stream_delay_ms=172),
        )
    )

    # 2. Servicios.
    vad = VADProcessor(
        vad_analyzer=FireRedVADAnalyzer(
            params=VADParams(
                # 0.2s (default) corta la frase en pedacitos con cualquier
                # micro-pausa/respiración natural. Se probó bajar a 0.4s
                # confiando en que smart-turn (ver services/jev_system1.py)
                # compensaría marcando INCOMPLETE los cortes falsos -- pero
                # smart-turn solo se evalúa DESPUÉS de que VAD ya decidió
                # "el usuario paró", nunca ve el resto de lo que la persona
                # iba a decir si VAD corta antes de tiempo. Confirmado en
                # vivo: cortes sistemáticos en pausas naturales de pensar
                # ("the socialism and [pausa buscando la palabra]"),
                # marcados COMPLETE porque hasta ahí sonaba terminado --
                # smart-turn no puede predecir audio que todavía no llegó.
                # Vuelto a 0.7s: la causa real de los cortes no era R2T2
                # (investigado a fondo, ver README) ni el grace period
                # (medido 0/12 salvados en inglés) -- era este timer
                # cortando antes de que la persona terminara de hablar.
                stop_secs=0.7,
            )
        )
    )  # Capa 0: VAD acústico (FireRedVAD streaming).
    # Nemotron 3.5 ASR (NVIDIA, cache-aware FastConformer-RNNT) corriendo
    # EN PROCESO en Windows -- ya no hace falta WSL2 ni un servidor
    # WebSocket aparte: eso era una restricción de R2T2 (vLLM no soporta
    # Windows nativo). Reemplazó a R2T2 tras medirlo en vivo contra las
    # mismas 14 frases (7 ES + 7 EN, ver README): R2T2 truncaba la última
    # palabra de forma consistente en ambos idiomas (reproducido también
    # con un clone 100% upstream sin modificar, así que no era config
    # nuestra), Nemotron no.
    nemotron_stt = NemotronASRService(
        # Español ahora es de primera clase: Nemotron lo lista como
        # "transcription-ready" y mide WER 4.11% en FLEURS-es, MEJOR que
        # su propio inglés (7.91%). La razón por la que el ASR estaba en
        # inglés era la debilidad de R2T2 en español (apuntaba a
        # chino/inglés) -- ya no aplica, y así deja de haber desajuste con
        # el LLM y la voz TTS, que ya estaban en español.
        language="Spanish",
    )
    # Smart-turn: modelo ONNX (viene empaquetado con pipecat, sin
    # descarga) que decide semánticamente si el usuario terminó de
    # hablar, en vez de contar un timer fijo de silencio. Reemplaza el
    # trade-off "timer corto = rápido pero corta palabras" / "timer largo
    # = preciso pero siempre lento" por una decisión real por turno.
    smart_turn = LocalSmartTurnAnalyzerV3()
    jev_router = JevSystem1Processor(
        stt=nemotron_stt, smart_turn=smart_turn, language="Spanish"
    )

    # Capa 3: System 2 (razonamiento). Groq (cloud, API compatible con
    # OpenAI, inferencia LPU muy rápida) para no competir por VRAM con el
    # servidor R2T2 en la GPU local.
    system2_context = build_shared_context(system_prompt=DEFAULT_SYSTEM_PROMPT)
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
    # jev_router: para que Jev sepa qué está diciendo el bot y pueda
    # distinguir su eco de una interrupción real del usuario (barge-in
    # sin palabra mágica -- ver _looks_like_bot_echo en jev_system1.py).
    system2_response_collector = System2ResponseCollector(system2_context, jev=jev_router)

    # WindowsTTSService (SAPI5 nativo, sin GPU -- ver services/windows_tts.py
    # para el detalle completo de por qué y cómo). Voz local "Dalia
    # (Natural)" en español, aunque R2T2/LLM siguen en inglés -- decisión
    # explícita del usuario: no vale la pena una voz en inglés no-HD
    # (Jenny/Dalia son la línea "Natural" vieja, no "Natural HD" como
    # Ava -- esa sí sonaba mejor pero solo existe como voz Online/cloud,
    # ver services/windows_tts.py). Mezcla de idiomas (conversación en
    # inglés, voz en español) es intencional, no un bug -- avisado al
    # usuario que es inusual, decisión suya igual.
    tts = WindowsTTSService(voice="Microsoft Dalia (Natural)")
    loopback_capture = WasapiLoopbackCapture(far_end_buffer)
    await loopback_capture.start()  # referencia far-end real (WASAPI loopback) para el AEC.

    # 3. Pipeline.
    pipeline = Pipeline(
        [
            transport.input(),  # Micro local (16 kHz) + AEC.
            vad,  # Capa 0: VAD acústico -> VADUserStarted/StoppedSpeakingFrame.
            nemotron_stt,  # Capa 1: ASR streaming cache-aware (Nemotron 3.5).
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
