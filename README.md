# Pipecat con audio local (Edge)

Asistente de voz conversacional, corriendo local en Windows con GPU (RTX 4090):

```
micrófono → FireRedVAD → Confucius4-R2T2 (ASR streaming, WSL2/vLLM)
  → Jev (System 1: reglas + acciones locales)
  → Groq (System 2: LLM en la nube, openai/gpt-oss-120b)
  → Kokoro-FastAPI (TTS, GPU CUDA) → altavoz/auriculares
```

Rama: `pipecat-local-audio-edge`.

## Arquitectura

- **VAD**: `services/firered_vad.py`, FireRedVAD streaming (confianza acústica,
  no solo energía). `stop_secs=0.7` para no cortar turnos a mitad de palabra.
- **ASR**: `services/r2t2_stt.py`, cliente WebSocket a un servidor R2T2 que
  corre en WSL2 (vLLM no tiene build nativo de Windows). Transcripción
  incremental (append-only deltas).
- **System 1** (`services/jev_system1.py`): reglas rápidas para acciones
  locales ("prende/apaga la luz", interrupciones) sin pasar por el LLM.
  Todo lo demás escala a System 2. Mutea el micrófono mientras el bot habla
  (+ 600ms de gracia) para evitar que se escuche a sí mismo.
- **System 2** (`services/system2_llm.py` + `OpenAILLMService`): Groq
  (`openai/gpt-oss-120b`), cloud, para no competir por VRAM con R2T2 (vLLM)
  en la misma GPU. Prompt corto/conversacional, `reasoning_effort="low"` +
  `max_completion_tokens=150` (ver "Notas" más abajo, por qué).
- **TTS**: `services/kokoro_gpu_tts.py`, cliente HTTP a un servidor
  Kokoro-FastAPI separado (PyTorch+CUDA real, no onnxruntime) corriendo en
  `127.0.0.1:8880`. Voz `ef_dora` (español).
- **AEC**: activo por defecto (`services/aec_filter.py`, WebRTC AEC3).
  La señal far-end (referencia de lo que suena por el parlante) se
  captura con WASAPI loopback real (`pyaudiowpatch`) en vez de tapear
  frames del pipeline de TTS -- el primer intento con eso tenía un delay
  far-end impredecible (colas internas de TTS) y degradaba el audio
  limpio. Con loopback el delay es chico y estable (buffers de
  hardware), permite usar parlantes en vez de auriculares sin que el
  sistema se re-transcriba a sí mismo.

## Arrancar (4 procesos: 3 obligatorios + observabilidad opcional)

### 1. Servidor R2T2 (WSL2, puerto 8272)

```bash
wsl -e bash -lc "export CPATH=/tmp/pydev/extracted/usr/include/python3.12:/tmp/pydev/extracted/usr/include:\$CPATH && \
  source ~/r2t2-venv/bin/activate && cd ~/Confucius4-R2T2 && \
  python -u ws_server.py --port 8272 \
    --asr_model_path /mnt/c/Users/<user>/.../sweetlips/models/Confucius4-R2T2 \
    --vad_model_path /mnt/c/Users/<user>/.../sweetlips/pretrained_models/FireRedVAD/Stream-VAD"
```

Listo cuando el log dice `model initialization complete`.

### 2. Servidor Kokoro-FastAPI (Windows, puerto 8880, GPU)

Setup e instrucciones de arranque: ver comentarios en `requirements.txt`
(sección "TTS: Kokoro-FastAPI"). Resumen del arranque, desde
`vendor/Kokoro-FastAPI`:

```powershell
$env:PHONEMIZER_ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
$env:USE_GPU = 'true'; $env:PYTHONUTF8 = '1'
$env:MODEL_DIR = 'src/models'; $env:VOICES_DIR = 'src/voices/v1_0'
$env:PYTHONPATH = '.'; $env:WEB_PLAYER_PATH = 'web'
.venv/Scripts/uvicorn.exe api.src.main:app --host 0.0.0.0 --port 8880
```

Listo cuando el log dice `Application startup complete` y
`Model warmed up on cuda: kokoro_v1`.

### 3. El pipeline (Windows)

```powershell
$env:GROQ_API_KEY = "gsk_..."   # https://console.groq.com/keys
pip install -r requirements.txt
python main.py
```

Listo cuando imprime `[Listo] El agente de voz Edge está escuchando...`.
Expone métricas Prometheus en `http://127.0.0.1:9091/metrics`.

### 4. Observabilidad (Prometheus + Grafana, opcional pero recomendado)

```powershell
cd observability
docker compose up -d
```

- Prometheus: http://localhost:9090 (target `voice-pipeline` -> debe estar `up`)
- Grafana: http://localhost:3000 (acceso anónimo habilitado como Viewer;
  admin/admin si querés editar) — dashboard **"Voice Pipeline - Latencia"**
  provisionado automático, sin pasos manuales.

El pipeline corre nativo en Windows (no en un container) y expone
`/metrics` en el puerto 9091; Prometheus (en Docker) lo scrapea vía
`host.docker.internal:9091` (ver `observability/prometheus.yml`).

## Observabilidad: qué mide el dashboard

`services/latency_probe.py` instrumenta cada turno que escala a System 2
con un histograma Prometheus (`voice_pipeline_stage_latency_seconds`,
label `stage`), medido desde que Jev decide escalar (t=0) hasta:

- `prompt enviado al LLM (Groq)`
- `LLM: primer token`
- `LLM: respuesta completa`
- `bot empieza a hablar (audio real)` — la métrica end-to-end que importa
  para "se siente conversacional o no".

El dashboard (`observability/grafana/provisioning/dashboards/voice-pipeline-latency.json`)
grafica p50/p95/p99 de cada etapa, un stat de p95 end-to-end de los
últimos 5 minutos, turnos escalados por ventana de 5 minutos, y si
Prometheus está scrapeando el pipeline (`up`/`down`). Para agregar una
etapa nueva: llamar `latency_probe.mark("nombre de la etapa")` en el
processor correspondiente — no hace falta tocar Prometheus/Grafana, el
label es dinámico.


## Notas / gotchas encontrados

- **`localhost` en Windows resuelve primero a IPv6 (`::1`)**, que no
  responde; las librerías HTTP/WS tardan ~2s en caer a IPv4 antes de
  conectar. Usar siempre `127.0.0.1` explícito (R2T2, Kokoro-FastAPI) —
  arregla también el "primer turno tarda 2-3s" que parecía ser del modelo.
- **`gpt-oss-120b` es un modelo "reasoning"**: sin `reasoning_effort="low"`,
  con un `max_completion_tokens` chico gasta casi todo pensando y devuelve
  respuestas de 1 palabra/letra. Con `reasoning_effort="low"` +
  `max_completion_tokens=150` responde bien y rápido (~150-400ms).
- **`OpenAITTSService` de pipecat no sirve para servidores TTS
  OpenAI-compatibles de terceros**: valida el nombre de voz contra una
  whitelist fija de voces de OpenAI y rechaza cualquier otra (p. ej.
  `ef_dora` de Kokoro). Por eso `services/kokoro_gpu_tts.py` es un wrapper
  propio en vez de usar la clase de pipecat directo.
- **`pipecat-ai` pineado a `1.9.0`**: la `1.11.0` tiene una regresión real
  donde el `StartFrame` nunca llega al final del pipeline, incluso en un
  pipeline mínimo de un solo processor.
- **XTTS-v2 descartado**: deprecado en pipecat 1.7+, licencia no comercial.
- **DirectML/CUDA vía onnxruntime descartados para TTS**: DirectML
  crashea con el op `ConvTranspose` de Kokoro; onnxruntime-gpu (CUDA) no
  encuentra `cublasLt64_13.dll` pese a tener los paquetes NVIDIA correctos
  instalados (bug de esa build de onnxruntime en Windows, no de instalación
  faltante). Por eso Kokoro-FastAPI (servidor PyTorch+CUDA aparte) en vez
  de intentar acelerar `pipecat-ai[kokoro]` (onnxruntime) por GPU.
- **`qwen/qwen3.8-27b` en vez de `openai/gpt-oss-120b`**: medido 3x más
  rápido (~220ms vs ~630ms) y sin tokens de razonamiento. Los modelos
  `gpt-oss` (20b y 120b) tienen un bug conocido de streaming (reportado
  en vLLM, LangChain, HF) donde el contenido de razonamiento rompe el
  parser de Groq ("Parsing failed") y el turno se pierde entero, mudo.
  `System2PromptBridge` además agrega un fallback hablado ante cualquier
  `ErrorFrame` del LLM (viaja río arriba, nunca llegaba al TTS antes).
- **AEC (WASAPI loopback) en modo bloqueante se cuelga para siempre en
  silencio total**: WASAPI no entrega paquetes de un endpoint de render
  idle. Solución: modo callback (PortAudio invoca solo cuando hay audio
  activo, que es justo el caso que importa).
- **Contención de GPU entre R2T2 (vLLM) y Kokoro-FastAPI**: ambos
  corren en la misma GPU física (RTX 4090). Con
  `gpu_memory_utilization=0.80` en R2T2, quedaba solo ~1.4GB libre
  (confirmado con `nvidia-smi`: 23.1/24.5GB en uso) y la síntesis de
  Kokoro variaba de forma errática entre 153ms y 2646ms para texto de
  largo similar -- no por la voz usada, sino por pelearse el turno de
  GPU con R2T2. Bajado a `gpu_memory_utilization=0.65` (editado en
  `~/Confucius4-R2T2/ws_server.py` dentro de WSL2, fuera de este repo)
  para dejarle ~5GB de margen a Kokoro. Si vuelven a aparecer picos de
  latencia erráticos en el TTS, revisar `nvidia-smi` primero antes de
  sospechar de la voz/idioma.

## Referencia: otros modelos ASR/TTS open-source (no usados, no aplica hoy)

Lista evaluada y descartada por ahora (no hay problema de precisión de
transcripción ni de calidad de voz reportado -- los problemas de esta
sesión fueron todos de timing de turno y de LLM, no de ASR/TTS):

- **ASR**: Hojo-ASR-Multi-V1 (multilenguaje, WER top), SenseVoiceSmall
  (234M, CPU, chino), whisper-large-v3-turbo (809M, 99 idiomas),
  parakeet-tdt-0.6b-v2 (NVIDIA, inglés), distil-large-v3.5 (756M).
  Nosotros usamos Confucius4-R2T2 (streaming, vLLM/WSL2) -- swap
  significaría re-hacer toda la integración WebSocket sin un problema
  real que lo justifique.
- **TTS**: ya elegimos Kokoro-82M (validado independientemente por esta
  lista como "el pequeño modelo que explota en inglés/multilenguaje"),
  corriendo en GPU vía Kokoro-FastAPI. Otras opciones si algún día hace
  falta clonación de voz o más idiomas: CosyVoice2-0.5B (zero-shot
  clone, Apache-2.0), Piper (15-28M, ultra liviano para edge/CPU).
- Reconsiderar sólo si aparece un problema real de precisión (nombres
  propios, ruido de fondo, idioma no soportado) -- no antes.
