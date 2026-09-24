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

import pyaudio


def _find_mic_device_index(name_substring: str, host_api_name: str = "MME") -> int:
    """Busca el índice del micrófono por NOMBRE + host API, no por índice
    fijo. Bug real encontrado en vivo hoy: `input_device_index=1` apuntaba
    a "Micrófono (Realtek USB Audio)" MME durante toda la sesión, pero
    tras instalar Equalizer APO (agrega endpoints virtuales al sistema) el
    orden de enumeración de PortAudio cambió y el índice 1 pasó a ser
    "Micrófono (Steren COM-126)" -- el mic de la webcam, el INCORRECTO
    (ya descartado hace tiempo). El pipeline quedó escuchando por la
    webcam sin ningún error visible -- silenciosamente mal, la peor clase
    de bug. Buscar por nombre es más lento (unas pocas ms al arrancar)
    pero sobrevive a que el índice vuelva a correrse."""
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] <= 0:
                continue
            host_api = p.get_host_api_info_by_index(info["hostApi"])["name"]
            if name_substring in info["name"] and host_api_name in host_api:
                return i
    finally:
        p.terminate()
    raise RuntimeError(
        f"No se encontró un micrófono con nombre que contenga {name_substring!r} "
        f"en host API {host_api_name!r}. Dispositivos disponibles cambiaron -- "
        f"revisar con pyaudio.PyAudio().get_device_info_by_index(i) para cada i."
    )


from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from services.firered_vad import FireRedVADAnalyzer
from services.jev_system1 import JevSystem1Processor
from services.windows_tts import WindowsTTSService
from services import latency_probe
from services.nemotron_stt import NemotronASRService
from services.wasapi_resampled_input import WASAPIResampledInputTransport
from services.system2_llm import (
    DEFAULT_SYSTEM_PROMPT,
    System2PromptBridge,
    System2ResponseCollector,
    build_shared_context,
)


async def main():
    latency_probe.start_metrics_server(port=9091)

    # 1. Audio local (micro y altavoz físicos del equipo).
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # Fijado por NOMBRE, no por índice numérico (ver
            # _find_mic_device_index más arriba): el equipo tiene DOS
            # micrófonos -- "Steren COM-126" (integrado a la webcam,
            # confirmado por su entrada duplicada como Camera en el
            # registro de dispositivos) y "Realtek USB Audio" (el externo
            # real, USB aparte). Bug real hoy: con índice fijo (=1), instalar
            # Equalizer APO agregó endpoints virtuales al sistema y corrió el
            # orden de enumeración de PortAudio -- el pipeline quedó
            # escuchando por la webcam sin ningún error visible. El orden
            # volvió a cambiar OTRA VEZ más tarde en la misma sesión (1 y 2
            # se intercambiaron de nuevo) -- por eso siempre por nombre.
            #
            # WASAPI (no MME): bug real medido en vivo hoy -- tras instalar
            # Equalizer APO, el backend MME quedó SILENCIADO para este
            # dispositivo (confirmado con captura cruda sin pipeline,
            # tools/wasapi_level_check.py: pico 35-44/32767 hablando fuerte,
            # A CUALQUIER TASA -- 16kHz forzado o 44100Hz nativo, mismo
            # resultado -- descartando que sea un problema de resampling).
            # WASAPI SÍ captura bien (pico 32502/32767, mismo hardware,
            # mismo momento). Se prescinde de EchoNull en sí (ver README
            # "Migración AEC": cancelaba la voz real, sin control de delay
            # expuesto) pero SE MANTIENE el transport WASAPI propio
            # (services/wasapi_resampled_input.py) porque es lo único que
            # captura bien tras la instalación de Equalizer APO -- WASAPI
            # nativo es 48kHz, se resamplea a 16kHz en proceso ahí mismo
            # (pedirle 16kHz directo a WASAPI tira "[Errno -9997] Invalid
            # sample rate"). El audio_in_filter (AEC3, ver abajo) SÍ se
            # sigue aplicando con este transport -- usa
            # `push_audio_frame()`, que alimenta la misma cola genérica de
            # `BaseInputTransport` donde pipecat aplica el filtro
            # (confirmado leyendo pipecat/transports/base_input.py).
            input_device_index=_find_mic_device_index("Realtek USB Audio", "WASAPI"),
            audio_in_sample_rate=16000,
            # AEC (WebRTC AEC3): DESACTIVADO por ahora (audio_in_filter=None).
            # Bug real hoy sin resolver: con stream_delay_ms=172 (calibrado
            # con MME) Nemotron recibía audio con RMS 3-4 (sobre-cancelado).
            # Con stream_delay_ms=0 (estimador interno de AEC3) el primer
            # turno funcionó bien, pero el segundo generó transcripción de
            # basura CONTINUA con el usuario en silencio real -- indica
            # inestabilidad del estimador, probablemente relacionada con
            # los warnings recurrentes "[AEC] Cola de loopback atrasada"
            # (el resampler/event loop de WasapiLoopbackCapture no sigue
            # el ritmo real del audio far-end, ver services/aec_filter.py).
            # Decisión: priorizar pipeline ESTABLE ahora (usuario con
            # auriculares, sin AEC) sobre seguir peleando esto en la misma
            # sesión. Ver README "Migración AEC" Capítulo 3 para retomar:
            # el jitter del loopback capture es el próximo sospechoso a
            # instrumentar antes de volver a intentar con AEC3 activo.
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
            #
            # max_completion_tokens subido de 55 a 400: bug real visto en
            # vivo -- 55 tokens alcanza apenas para una frase corta en
            # español (más tokens/palabra que en inglés), y el modo
            # [DETAILED_ANSWER] (slow path, ver jev_system1.py) pide
            # explícitamente respuestas de varias frases -- el límite fijo
            # las cortaba a mitad de explicación sin importar lo que diga
            # el prompt (confirmado: "Depende mucho de a qué te
            # refieras..." y "La diferencia fundamental entre filosofía e
            # ideología..." truncadas). El prompt ya instruye "una frase
            # corta" para el modo normal -- el modelo para solo (finish
            # reason "stop") mucho antes de 400 en ese caso; el tope solo
            # importa como techo de seguridad, no como control primario
            # de longitud.
            model="qwen/qwen3.8-27b",
            max_completion_tokens=400,
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
    # (Natural)" en español -- ASR (Nemotron), LLM y TTS están los tres en
    # español desde la migración a Nemotron (antes el ASR estaba forzado a
    # inglés por la debilidad de R2T2 ahí; ya no aplica, Nemotron mide
    # mejor en español que en inglés).
    tts = WindowsTTSService(voice="Microsoft Dalia (Natural)")

    # WASAPIResampledInputTransport en vez de transport.input(): ver
    # comentario en input_device_index más arriba -- MME quedó roto tras
    # instalar Equalizer APO, WASAPI es lo único que captura bien.
    # loopback_capture (WasapiLoopbackCapture) NO se arranca -- solo hacía
    # falta como referencia far-end para audio_in_filter, que está
    # desactivado por ahora (ver comentario arriba).
    mic_input = WASAPIResampledInputTransport(
        transport._pyaudio, transport._params, native_rate=48000
    )

    # 3. Pipeline.
    pipeline = Pipeline(
        [
            mic_input,  # Micro local (WASAPI 48kHz -> 16kHz). AEC desactivado, usar auriculares.
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

    # setup_timeout_secs subido de 20s (default pipecat) a 60s, y hoy a
    # 150s: segundo bug real visto en vivo -- tras muchos reinicios
    # seguidos del pipeline en la misma sesión (7+), una corrida de carga
    # de Nemotron que normalmente tarda 5-7s tardó más de 60s (confirmado
    # con el log: "[Nemotron] Modelo cargado" llegó 26s DESPUÉS de que el
    # timeout ya había tirado abajo el pipeline). VRAM y RAM verificadas
    # sanas en ese momento (nvidia-smi: 1.5/24.5GB; RAM: 18/31.7GB libres)
    # -- no es contención de memoria, probablemente variabilidad de I/O de
    # disco/SO tras una sesión larga con muchos reinicios de modelos
    # pesados. 150s da margen real sin ocultar un cuelgue genuino (si
    # tarda más que eso, sí hay algo mal).
    task = PipelineTask(
        pipeline, enable_rtvi=False, idle_timeout_secs=None, setup_timeout_secs=150.0
    )
    runner = PipelineRunner()

    print("\n[Listo] El agente de voz Edge está escuchando... (Ctrl+C para salir)\n")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
