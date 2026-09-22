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
- **AEC**: implementado (`services/aec_filter.py`, WebRTC AEC3) pero
  **deshabilitado** por defecto (`audio_in_filter=None` en `main.py`) —
  degradaba la señal limpia en varios intentos. Usar auriculares en vez de
  parlantes evita el problema de raíz sin necesitar AEC.

## Arrancar (3 procesos)

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
