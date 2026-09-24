# Pipecat con audio local (Edge)

Asistente de voz conversacional, corriendo local en Windows con GPU (RTX 4090):

```
micrófono → FireRedVAD → Nemotron 3.5 ASR (streaming cache-aware, en proceso)
  → Jev (System 1: reglas + acciones locales)
  → Groq (System 2: LLM en la nube, openai/gpt-oss-120b)
  → Windows TTS (SAPI5, voz "Dalia (Natural)") → altavoz/auriculares
```

Rama: `pipecat-local-audio-edge`.

> ✅ **ESTADO ACTUAL: funcional y estable SIN AEC, parado a distancia del parlante.**
> Ver sección "Migración AEC" más abajo, Capítulo 3, para el detalle completo y
> qué falta para volver a usar el parlante de cerca. Resumen para retomar rápido:
> - MME (input de mic clásico) quedó **roto/silenciado** tras instalar Equalizer APO
>   (medido: pico 35-44/32767 hablando fuerte, a cualquier tasa) -- no es reversible
>   con más config, el mic usa **WASAPI** de ahora en más
>   (`services/wasapi_resampled_input.py`).
> - EchoNull (AEC por GPU, Equalizer APO): **descartado**, cancela la voz real casi
>   al 100%, sin control de delay expuesto.
> - AEC3 casero (`WebRTCAECFilter`) + WASAPI: **desactivado por ahora**
>   (`audio_in_filter=None` en `main.py`) -- con `stream_delay_ms` fijo (172ms,
>   calibrado para MME) sobre-cancelaba (RMS 3-4, Nemotron no transcribía nada);
>   con `stream_delay_ms=0` (estimador automático) funcionó un turno y después
>   generó transcripción de basura continua en silencio real -- inestabilidad
>   sin resolver, sospecha: jitter del loopback capture (ver logs
>   `[AEC] Cola de loopback atrasada`).
> - **Verificado en vivo, sesión completa (36 turnos, conversación larga y
>   natural)**: funciona bien SIN auriculares, parado a **más de 1 metro** del
>   parlante de la compu -- menos acoplamiento acústico directo (factor real
>   medible, no solo software). Sin AEC el mic sí capta algo de eco del propio
>   bot (confirmado: transcripción palabra por palabra sincronizada con "Bot
>   started/stopped speaking"), pero el filtro de eco por SOFTWARE de Jev
>   (`_looks_like_bot_echo`, compara texto contra lo último que dijo el bot)
>   lo descarta correctamente -- nunca escaló eco como si fuera el usuario.
>   Usuario confirma: "por primera vez lo sentí fluido, aun con algo de ruido
>   lejano de cortas pausas".
> - Sin AEC: cuanto más cerca del parlante, más eco real llega al mic y más
>   presión sobre el filtro de software (probado funcionando >1m; no probado
>   de cerca). Requiere distancia del parlante O auriculares hasta retomar
>   el AEC de verdad.

## Arquitectura

- **VAD**: `services/firered_vad.py`, FireRedVAD streaming (confianza acústica,
  no solo energía). `stop_secs=0.7` para no cortar turnos a mitad de palabra.
- **ASR**: `services/nemotron_stt.py`, Nemotron 3.5 ASR de NVIDIA (600M,
  cache-aware FastConformer-RNNT) corriendo EN PROCESO vía 🤗 Transformers.
  Transcripción incremental por deltas, ~2.9GB VRAM, español "tier 1"
  (WER 4.11% en FLEURS-es, mejor que su propio inglés). Reemplazó a
  Confucius4-R2T2 tras medirlo (ver "Migración ASR" más abajo).
- **System 1** (`services/jev_system1.py`): reglas rápidas para acciones
  locales ("prende/apaga la luz", interrupciones) sin pasar por el LLM.
  Todo lo demás escala a System 2. Mutea el micrófono mientras el bot habla
  (+ 600ms de gracia) para evitar que se escuche a sí mismo.
- **System 2** (`services/system2_llm.py` + `OpenAILLMService`): Groq
  (`openai/gpt-oss-120b`), cloud, para no competir por VRAM con el ASR local
  en la misma GPU. Prompt corto/conversacional, `reasoning_effort="low"` +
  `max_completion_tokens=150` (ver "Notas" más abajo, por qué).
- **TTS**: `services/windows_tts.py`, SAPI5 nativo de Windows con las voces
  "Natural" desbloqueadas vía NaturalVoiceSAPIAdapter. Voz `Microsoft Dalia
  (Natural)` (español, local, ~147ms). Cero GPU, cero red.
  `services/kokoro_gpu_tts.py` queda como alternativa si se quiere volver.
- **AEC**: DESACTIVADO actualmente (`services/aec_filter.py`, WebRTC AEC3,
  `audio_in_filter=None` en `main.py` -- ver banner de estado arriba y
  "Migración AEC" Capítulo 3 más abajo para el detalle completo). El código
  sigue en el repo, listo para reactivar (`audio_in_filter=WebRTCAECFilter(...)`)
  cuando se resuelva la inestabilidad con el nuevo transport WASAPI. Mientras
  tanto: mic vía `services/wasapi_resampled_input.py` (WASAPI, no MME -- ver
  banner), sin cancelación de eco acústico -- el filtro de eco por texto en
  `services/jev_system1.py` (`_looks_like_bot_echo`) es la única defensa
  activa contra que el bot se escuche a sí mismo.

## Arrancar (1 proceso obligatorio + observabilidad opcional)

Ya no hace falta WSL2 ni un servidor de TTS aparte: el ASR (Nemotron) corre
en proceso y el TTS usa SAPI5 nativo de Windows. Todo vive en `main.py`.

### 1. El pipeline (Windows)

```powershell
$env:GROQ_API_KEY = "gsk_..."   # https://console.groq.com/keys
pip install -r requirements.txt
python main.py
```

La primera corrida descarga el checkpoint de Nemotron (~1.2GB) al cache de
Hugging Face; las siguientes cargan en ~5s. Listo cuando imprime
`[Listo] El agente de voz Edge está escuchando...`.
Expone métricas Prometheus en `http://127.0.0.1:9091/metrics`.

### 2. (Opcional) Kokoro-FastAPI, solo si se vuelve a ese TTS

Setup e instrucciones: ver comentarios en `requirements.txt` (sección
"TTS: Kokoro-FastAPI"). Los smoke tests de `tools/` también lo usan para
sintetizar frases de prueba. Desde `vendor/Kokoro-FastAPI`:

```powershell
$env:PHONEMIZER_ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
$env:USE_GPU = 'true'; $env:PYTHONUTF8 = '1'
$env:MODEL_DIR = 'src/models'; $env:VOICES_DIR = 'src/voices/v1_0'
$env:PYTHONPATH = '.'; $env:WEB_PLAYER_PATH = 'web'
.venv/Scripts/uvicorn.exe api.src.main:app --host 0.0.0.0 --port 8880
```

### 3. Observabilidad (Prometheus + Grafana, opcional pero recomendado)

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


## Migración ASR: Confucius4-R2T2 → Nemotron 3.5 ASR

**Motivo**: R2T2 truncaba la última palabra de una fracción grande de los
turnos. Se investigó a fondo antes de migrar; vale la pena leer los
resultados negativos para no repetir el trabajo.

### Qué se descartó primero (el bug NO era config nuestra)

- Timeout de `flush_final()` (0.6s → 1.2s): ayuda, no resuelve.
- Silencio de cola extra antes del EOS: no resuelve.
- Presupuesto de tokens del servidor (`first_max_new_tokens` 4 → 16 en el
  `ws_server.py` de R2T2): **efecto cero**, revertido.
- `stop_secs` del VAD (0.4 → 0.7s): mejora real y se mantiene, pero el
  truncamiento seguía apareciendo igual.
- **Clone 100% upstream**: se clonó el repo oficial de R2T2 limpio, con los
  mismos pesos y el mismo FireRedVAD, y **reprodujo el bug idéntico**
  (`"privada"` → `"priv"`, `"es"` y `"luz"` perdidas enteras). La única
  diferencia con nuestra copia era `gpu_memory_utilization`, que no afecta
  precisión. Conclusión: es comportamiento del modelo, no configuración.

### Comparación medida (14 frases: 7 ES + 7 EN, misma síntesis y mismo audio)

| Motor | Truncamientos | VRAM | Streaming real | Español |
| --- | --- | --- | --- | --- |
| Confucius4-R2T2 (1.7B, WSL2/vLLM) | **6/14** | ~15.6GB | sí (frágil) | secundario |
| Parakeet TDT v3 (0.6B int8, CPU) | 0 | 0 | **no** (export offline) | ok |
| Nemotron 3.5 ASR (0.6B, GPU fp32) | 0 | ~2.9GB | **sí** (cache-aware) | **tier 1** |
| Nemotron 3.5 ASR (0.6B GGUF, CPU) | 1 pérdida total | 0 | sí | tier 1 |

Se eligió **Nemotron GPU sin cuantizar**: la variante GGUF/CPU ahorra VRAM
pero la cuantización introdujo una falla nueva (perdió una frase corta
entera), y la prioridad del proyecto es precisión sobre latencia/recursos.

Por qué R2T2 muestra buen WER en sus benchmarks y aun así falla acá: sus
datasets (AMI, LibriSpeech, GigaSpeech...) son habla continua larga, donde
casi nunca hay que decidir si comprometer la última palabra con poco
contexto futuro. El caso conversacional real -- turnos cortos que terminan
justo al final de la idea -- casi no se ejercita ahí. Es exactamente lo que
señala el paper *RW-Voice-EQ Bench* de Hume AI (arXiv:2607.14846): *"real
world ... conversational conditions expose failures that are not captured by
established clean-speech benchmarks"*.

### Bug de integración encontrado al portar (vale documentarlo)

El bridge de streaming perdía la última palabra de CADA turno, pero solo con
audio en tiempo real (con el audio precargado de golpe funcionaba). Tres
hipótesis fallidas antes de dar con la causa: alineación de chunks STFT,
`prompt_ids` faltante en `generate()`, padding del chunk final.

**Causa real**: `loop.call_soon_threadsafe()` solo **agenda** el callback en
el event loop, no lo ejecuta. `flush_final()` cerraba el turno (bloqueante) y
drenaba la cola de texto **sin ceder nunca el control al event loop**, así
que los callbacks con el último token seguían pendientes. Con audio
precargado no se notaba porque los tokens llegaban mucho antes, entre los
`await` del loop de entrada. Fix: `await asyncio.sleep(0.05)` después de
cerrar el turno, y correr el cierre (que hace `thread.join()`) en un executor
para no bloquear el loop.

Verificación: `python tools/smoke_nemotron_multi.py` → 4/4 frases exactas en
ES y EN, deltas confirmados incrementales durante el turno, y la reapertura
de turno probada (segundo turno seguido con la misma instancia también OK).

## Migración AEC: WebRTC AEC3 casero → EchoNull → WASAPI+AEC3 → AEC desactivado (funcional)

**Motivo**: el AEC casero (`services/aec_filter.py`, WebRTC AEC3 + loopback
WASAPI manual) seguía amplificando en vez de cancelar en ~15-20% de los
picos de volumen, incluso después de 3 rondas de fixes reales (buffer
desalineado 4.4s, underrun al bajar el tope, delay sin calibrar). Se
evaluaron las alternativas y se decidió migrar a una solución de sistema
operativo en vez de seguir iterando en la nuestra.

### Por qué no había alternativa madura dentro de pipecat

Los filtros de audio que trae pipecat (`KoalaFilter`, `RNNoiseFilter`,
`KrispVivaFilter`, `AICFilter`) son todos de **supresión de ruido**, no de
**cancelación de eco** -- no toman una señal de referencia far-end, así que
no pueden hacer lo que hace un AEC. WebRTC AEC3 (via `pywebrtc-audio`) era
la única opción de cancelación de eco real disponible, y expone una API
mínima (sample_rate, num_channels, stream_delay_ms, process, reset) sin
ninguna visibilidad de diagnóstico -- todo el trabajo de hoy fue
reverse-engineering su comportamiento a ciegas.

### La alternativa elegida: EchoNull

[EchoNull](https://github.com/Skyline-23/EchoNull) (MIT, 0 stars/forks --
riesgo de baja adopción aceptado explícitamente) integra el modelo de AEC
de NVIDIA (NvAFX, red neuronal, corre en GPU) como plugin VST dentro de
[Equalizer APO](https://sourceforge.net/projects/equalizerapo/) (framework
de procesamiento de audio de Windows, maduro, ampliamente usado). Corre
**a nivel de sistema operativo**, en `audiodg.exe` -- limpia el audio del
micrófono ANTES de que llegue a cualquier proceso, incluido el nuestro.

### Instalación (pasos manuales, no automatizables del todo)

1. Instalar Equalizer APO (`EqualizerAPO-x64-1.4.2.exe` desde SourceForge).
   En el "Device Selector": marcar el dispositivo de PLAYBACK real (el que
   aparece como "Default device") y el mic real en CAPTURE (`Micrófono /
   Realtek USB Audio` en este equipo -- NO el que dice Steren COM-126, es
   la webcam).
2. **Reiniciar Windows** -- obligatorio, no opcional. El primer intento sin
   reiniciar tiró "This application failed to start because no Qt platform
   plugin could be initialized" al abrir `Editor.exe`.
3. Instalar EchoNull (`EchoNullSetup-Ada-RTX40.exe` para RTX 40 series --
   hay builds separados por arquitectura NVIDIA, verificar SHA-256 contra
   el `.sha256` publicado en el release antes de correr cualquier instalador
   de un proyecto de baja adopción).
4. Si el Qt Platform Plugin sigue fallando después de reiniciar: bug de
   empaquetado real encontrado hoy -- los DLLs `Qt6*.dll` quedan en la raíz
   de `C:\Program Files\EqualizerAPO\`, pero `qwindows.dll` (el plugin de
   plataforma) queda en `qt\platforms\`, una ruta que Qt no busca por
   default. Fix: copiar (como administrador) `qt\platforms\*` a
   `C:\Program Files\EqualizerAPO\platforms\`.
5. Abrir `Editor.exe`, ir a la sección de CAPTURE, abrir el panel de
   `EchoNullPlugin` ("Open panel"), elegir el dispositivo de reproducción
   real como "Playback reference", y **APPLY REFERENCE**.

### El obstáculo real: MME vs WASAPI

Equalizer APO/EchoNull solo intercepta streams **WASAPI** reales -- el mic
vía **MME** (lo que usaba el pipeline hasta hoy, porque resamplea
automático a 16kHz) no pasa por ese pipeline de audio compartido, así que
el panel seguía mostrando "AUDIO ENGINE OFFLINE" con el pipeline corriendo.
Probar WASAPI directo con `audio_in_sample_rate=16000` falla con
`[Errno -9997] Invalid sample rate`: el dispositivo nativo es 48kHz y
PortAudio no resamplea automático para ese backend (sí lo hace para MME).

Un `audio_in_filter` (el mecanismo que usaba el AEC casero) **no alcanza**
para resolver esto: el filtro solo puede tocar `frame.audio` (los bytes),
pero `frame.sample_rate` ya quedó fijado en 16000 antes de que el filtro
corra -- resamplear ahí desalinea el frame (dice 16000Hz pero lleva menos
muestras de las que corresponden).

**Fix real**: `services/wasapi_resampled_input.py`
(`WASAPIResampledInputTransport`, hereda de
`pipecat.transports.local.audio.LocalAudioInputTransport`) abre el stream
PyAudio a la tasa NATIVA (48kHz) y resamplea cada chunk a 16kHz en el
propio callback antes de armar el `InputAudioRawFrame` -- así el frame
reporta el sample_rate correcto Y el contenido corresponde de verdad a esa
tasa. `LocalAudioTransport.input()` no permite inyectar un transport propio
(hardcodea la clase), así que se instancia esta clase directo en la lista
del pipeline, compartiendo el mismo `pyaudio.PyAudio()` que
`transport.output()` (parlante) para no abrir el subsistema de audio dos
veces.

Verificado en vivo: panel de EchoNull pasó de "AUDIO ENGINE OFFLINE" a
**"RTX AEC ACTIVE"** con el pipeline corriendo de verdad.

### Resultado: EchoNull cancelaba la voz real (no era tuning)

Con el transport WASAPI andando y el panel confirmando "RTX AEC ACTIVE",
la prueba en vivo dio transcripciones de basura ("Olana Spraylo ne podemos
tu Quelo Bueno" en vez de "hola, ¿me escuchás?") y el pipeline se quedaba
colgado (VAD nunca veía silencio real). Se aisló la causa con una medición
cruda del nivel de señal (`tools/wasapi_level_check.py`, RMS/pico crudo del
dispositivo, sin pasar por el pipeline):

| Config EchoNull | Pico máx. capturado (de 32767) |
| --- | --- |
| AEC ON, reference strength máximo | ~32 |
| AEC ON, reference strength bajo | ~32 (igual) |
| **AEC OFF** | **32502** (normal) |

**EchoNull cancelaba la voz real del usuario casi al 100%, independiente
de la fuerza configurada** -- no era un tema de tuning. Se confirmó que el
dispositivo de referencia de reproducción SÍ era el correcto (coincide con
el output real del pipeline, `FHQD40IF01`, verificado con
`get_default_output_device_info()`). La causa más probable: EchoNull no
expone NINGÚN control de delay/alineación temporal en su UI (solo strength
y on/off) -- sin eso, un AEC (clásico o neuronal) queda desalineado y
entra en falsos positivos masivos, cancelando audio que no tiene relación
real con el playback. Nuestro AEC3 evita esto porque el delay SÍ está
calibrado con medición real (chirp, 172ms).

### Capítulo 2: MME se rompió (bug distinto, no relacionado a EchoNull)

Se revirtió el mic a MME + `WebRTCAECFilter` (el AEC3 casero que ya
funcionaba antes de tocar nada de esto). Pero midiendo en vivo con
`tools/wasapi_level_check.py` (captura cruda, sin pasar por el pipeline)
se encontró que **MME quedó silenciado para este dispositivo tras instalar
Equalizer APO**, a cualquier tasa:

| Backend | Tasa | Pico máx. hablando fuerte (de 32767) |
| --- | --- | --- |
| MME | 16kHz forzado | 35 |
| MME | 44100Hz nativo | 44 |
| **WASAPI** | 48kHz nativo | **32502** |

Mismo hardware, mismo momento, mismo usuario. MME (el backend legacy que
usaba el pipeline desde el principio) quedó roto por la instalación de
Equalizer APO -- no es un problema de resampling (se probó a tasa nativa,
mismo resultado), ni de volumen de Windows (confirmado al 100%/máximo).
WASAPI es la única vía utilizable ahora en este equipo.

### Capítulo 3: WASAPI + AEC3 combinados, Nemotron no transcribe nada (bug encontrado)

Se combinó `WASAPIResampledInputTransport` (mic a WASAPI, resampleado a
16kHz en proceso) con `audio_in_filter=WebRTCAECFilter` (nuestro AEC3,
igual que antes) -- confirmado por lectura del código fuente de pipecat
(`pipecat/transports/base_input.py`) que el filtro SÍ se aplica igual con
este transport custom (usa `push_audio_frame()`, la misma cola genérica
de `BaseInputTransport` donde se invoca `audio_in_filter.filter()`).

El pipeline arranca sin errores, el AEC inicializa bien
(`[AEC] EchoCanceller iniciado`), el diagnóstico periódico confirma señal
de nivel normal llegando (`near_rms` visto entre 21 y 961 en distintos
intentos, nada anormalmente bajo), y VAD detecta correctamente
`User started/stopped speaking`. **Pero Nemotron nunca produce ni un solo
carácter de texto** -- cero líneas `[Jev] escuchado` en turnos completos,
confirmado en al menos 3 intentos separados con señal de nivel razonable.
El watchdog de turno atascado (15s) eventualmente descarta el turno con
texto vacío (`"..."`).

**Diagnóstico real (no hipótesis)**: se instrumentó
`NemotronASRService.run_stt()` con un log directo de tamaño+RMS de cada
chunk recibido (`_diag_counter`, cada 25 llamadas). Confirmó que Nemotron
SÍ recibía audio, pero con **RMS 3-4** (de 32767) -- coincidía exactamente
con `cleaned_rms` del AEC en los mismos instantes. **El AEC3 estaba
sobre-cancelando la señal real**, no un problema de que el audio no
llegara.

Causa: `stream_delay_ms=172` (calibrado con `tools/calibrate_aec_delay.py`
cuando el mic era MME) quedó **obsoleto** -- el camino cambió a WASAPI +
resample async propio (`services/wasapi_resampled_input.py`), que agrega
latencia de cola/scheduling nueva del lado del mic que no existía con MME.
Con el delay desalineado, AEC3 resta la señal equivocada del audio real
en vez del eco -- exactamente el comportamiento documentado ya en el
propio código (`WebRTCAECFilter.__init__`, ver "firma clásica de un
filtro adaptativo con el delay desalineado").

### Capítulo 4: `stream_delay_ms=0` (estimador automático) funciona un turno, después se desestabiliza

Cambiar a `stream_delay_ms=0` (estimador interno de AEC3, en vez del
valor fijo obsoleto) **resolvió la sobre-cancelación**: verificado en vivo,
un turno completo funcionó de punta a punta (transcripción limpia,
escalada a LLM, respuesta de Groq, TTS, audio real).

Pero el turno siguiente generó **transcripción de basura CONTINUA con el
usuario en silencio real confirmado** ("Saluda usar tumo hablando
naturalmente bien aquí siendo los fríos sistemas..." -- sin sentido,
acumulando sin parar, nunca cierra el turno). Esto no es un problema de
volumen -- es inestabilidad activa: el sistema genera contenido de la
nada. Sospecha, sin confirmar: los warnings recurrentes
`[AEC] Cola de loopback atrasada` (el resampler/event loop de
`WasapiLoopbackCapture` no sigue el ritmo real del audio far-end) podrían
estar desestabilizando el estimador de delay automático, que necesita una
señal de referencia consistente para converger.

### Capítulo 5 (ACTUAL): AEC desactivado, verificado funcional y estable en vivo

Dada la inestabilidad del Capítulo 4, se desactivó el AEC3 por completo
(`audio_in_filter=None`). Verificado en vivo en una sesión larga (36
turnos, conversación natural sostenida sobre temas variados) parado a
**más de 1 metro** de distancia del parlante de la PC (sin auriculares):
funciona bien, sin cuelgues ni transcripción de basura. Usuario: *"por
primera vez lo sentí fluido, aun con algo de ruido lejano de cortas
pausas"*.

El mic SÍ capta algo del audio del propio bot mientras habla (confirmado:
transcripción palabra por palabra en `[Jev] escuchado` sincronizada
exactamente con `Bot started/stopped speaking`), pero el filtro de eco
por **software** de Jev (`_looks_like_bot_echo` en
`services/jev_system1.py`, compara el texto escuchado contra lo último
que dijo el bot) lo descarta correctamente en todos los casos observados
-- nunca se escaló eco al LLM como si fuera el usuario. La distancia al
parlante (menos acoplamiento acústico directo) es un factor real que
contribuye; no se probó qué tan bien aguanta hablando pegado al parlante.

**Estado para retomar el AEC** (no urgente, el pipeline funciona sin él si
se mantiene distancia del parlante o se usan auriculares):
1. Instrumentar `WasapiLoopbackCapture._drain()` para entender por qué el
   resampler/event loop no sigue el ritmo del far-end (causa raíz
   sospechada de la inestabilidad del Capítulo 4).
2. Una vez estable el loopback, recalibrar `stream_delay_ms` con un
   chirp que replique el camino COMPLETO real (WASAPI nativo + resample
   async), no solo MME directo como hace hoy
   `tools/calibrate_aec_delay.py`.
3. Alternativa a considerar: mover el resample del mic fuera del hilo
   async (ej. resamplear síncrono en el propio callback de PyAudio antes
   de despachar) para eliminar la fuente de jitter en vez de perseguirla.


### Capítulo 6: bug no relacionado encontrado en el camino (mic roto silenciosamente)

Instalar Equalizer APO agregó endpoints de audio virtuales al sistema, lo
que corrió el ORDEN DE ENUMERACIÓN de PortAudio -- `input_device_index=1`
(fijo, hardcodeado, usado toda la sesión) pasó de apuntar a "Micrófono
(Realtek USB Audio)" a apuntar a "Micrófono (Steren COM-126)" (el mic de
la webcam, ya descartado como incorrecto hace tiempo). El pipeline quedó
escuchando por la webcam **sin ningún error visible** -- la peor clase de
bug, y probablemente explica (al menos en parte) la calidad de
transcripción muy degradada observada en un tramo de la sesión de hoy.

Fix: `_find_mic_device_index()` en `main.py` busca el micrófono por NOMBRE
+ host API en vez de por índice numérico fijo. Instalar/desinstalar
cualquier dispositivo o driver de audio (como pasó hoy) puede volver a
correr los índices -- buscar por nombre sobrevive a eso.

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
- **Contención residual de CÓMPUTO (no memoria), confirmada con
  `nvidia-smi dmon` durante turnos reales**: incluso con VRAM sana
  (~5GB libres), R2T2 mantiene la GPU con SM utilization oscilando
  20-55% de forma CONTINUA mientras procesa el stream de audio en
  tiempo real (no solo en picos por turno) -- Kokoro sintetizando en
  ese momento mide 324-703ms en vez de los 170-300ms aislados (medido
  con `curl` directo a Kokoro sin R2T2 activo, para descartar Docker/
  observability-stack como causa -- confirmado que NO son el problema,
  tiempos iguales con contenedores parados o corriendo). Es contención
  de cómputo, no de memoria: dos modelos de inferencia real-time en la
  misma GPU van a competir por ciclos aunque sobre VRAM. No hay fix
  fácil sin separar las cargas en GPUs distintas -- aceptado como costo
  estructural de la arquitectura en cascada (mismo trade-off que la
  decisión de no migrar a un modelo speech-to-speech unificado).
- **Truncamiento residual de la última palabra en R2T2, ~40% determinístico
  (confirmado con `tools/test_r2t2_truncation.py`)**: script de
  reproducción (sintetiza frases con Kokoro, las manda directo a R2T2 por
  WebSocket, sin pipecat ni micrófono real) muestra que 3 de 7 frases de
  prueba pierden la última palabra SIEMPRE, en 3+ corridas idénticas --
  no es ruido aleatorio ("qué hora es" -> "qué hora", "prendé la luz" ->
  "prende la", "...propiedad privada" -> "...propiedad priv"). Confirmado
  que NO es un bug de nuestro pipeline: revisado el código fuente de
  `STTService` (pipecat) -- `process_audio_frame()` no gatea por estado
  de VAD, el audio sigue fluyendo a R2T2 continuo durante toda la ventana
  de gracia antes de `flush_final()`. Tampoco es el timeout de
  `flush_final()` (ya subido a 1.2s) ni la falta de silencio de cola (el
  test agrega 0.5-1.0s, mismo resultado). **Hipótesis descartada
  activamente**: `first_max_new_tokens` en el manejo del EOS de
  `ws_server.py` (`asr_stream_api_v1`, línea ~1007) es
  `max(1, (step+lookahead)/1280) = 4` -- subido a `max(16, ...)` en vivo,
  reiniciado el servidor (~5min), re-corrido el test: **mismo resultado
  exacto, ningún cambio**. Revertido. El cuello de botella no es
  presupuesto de tokens de generación -- probablemente algo en cómo se
  codifica/pierde el audio de la última palabra antes de llegar a esa
  etapa (dentro de `asr_model.init_streaming_state`/
  `finish_streaming_transcribe_no_reset`, código del paquete del modelo,
  no de `ws_server.py`). Sin investigar más a fondo por hoy -- afecta
  desproporcionadamente frases cortas de 2-3 palabras, que son
  justo el patrón típico de las acciones locales (hora/luces).


- **Router fast/slow (patrón portado de FXPerto `QueryRouter`)**: Jev ya
  no manda todo por el mismo presupuesto de latencia. `_is_slow_path()`
  en `services/jev_system1.py` clasifica el texto escalado con una
  heurística sin costo (largo >= 12 palabras, o keywords tipo "explain
  in detail"/"compare"/"walk me through"), sin llamada extra al LLM
  (0ms). Si es slow path: Jev empuja un chime no hablado (dos notas
  ascendentes, `OutputAudioRawFrame` generado con `numpy`, 180ms) --
  audio crudo en vez de un ack sintetizado por TTS: suena en cuanto se
  empuja el frame, sin esperar síntesis, y como no es
  `TTSAudioRawFrame` no dispara `BotStartedSpeakingFrame` (no hace
  falta silenciar el ASR por un sonido tan corto). El `TextFrame` real
  lleva el prefijo `[DETAILED_ANSWER]`, que el system prompt
  (`DEFAULT_SYSTEM_PROMPT`/`_EN` en `services/system2_llm.py`)
  interpreta como permiso para saltarse la regla de "una frase". No
  hace falta `asyncio.create_task` ni gestión de background task: los
  frames de pipecat ya son async, así que el chime suena mientras el
  LLM arma la respuesta larga en paralelo. Motivación: no tiene sentido
  perseguir <250ms para TODAS las consultas -- las simples ya están en
  ~400-500ms (bien), y las complejas dejan de competir por ese
  presupuesto porque el usuario ya sabe que "está pensando" (con un
  sonido, no una frase hablada -- menos intrusivo, más rápido).

- **TTS nativo de Windows (SAPI5) en vez de Kokoro-FastAPI**: investigado
  como alternativa sin GPU para eliminar de raíz la contención de
  cómputo entre R2T2 y Kokoro (ver arriba). Windows trae voces "OneCore"
  clásicas (David/Zira/Mark en inglés, Raul/Sabina en español) gratis y
  locales, pero no se ven vía SAPI5 clásico (`SAPI.SpVoice`/
  `System.Speech`) por defecto -- solo vía la API WinRT
  `Windows.Media.SpeechSynthesis`, cuyo binding de Python en PyPI
  (`winrt-Windows.Media.SpeechSynthesis` 3.2.1) no expone `AllVoices`
  (confirmado revisando el binding nativo) así que no se puede elegir
  voz por idioma con ese paquete. Solución: `tools/add-onecore-voices.ps1`
  copia las claves de registro de OneCore al namespace clásico de SAPI5
  (aportado por el usuario, basado en
  https://github.com/microsoft/VibeVoice) -- con eso, `SAPI.SpVoice`
  ve las 5 voces y sí permite seleccionar por nombre. Medido: 30-61ms,
  sin GPU.
  - Las voces "Natural" de Windows 11 (Ava, Jenny, Aria...) suenan mucho
    mejor, pero Microsoft las bloquea a propósito para apps de
    terceros: confirmado leyendo `AppxManifest.xml` del paquete
    instalado -- están registradas como `windows.appExtension` tipo
    `com.microsoft.voice.model.1`, consumible solo por el propio
    Narrador, no por `AllVoices` ni SAPI5. Documentado independientemente
    en Wikipedia/GitHub: *"as of 2024, no other third-party applications
    are able to use these voices in any way, shape or form"*.
  - Desbloqueadas de todos modos con
    [NaturalVoiceSAPIAdapter](https://github.com/gexgd0419/NaturalVoiceSAPIAdapter)
    (955★, MIT) -- extrae claves de cifrado de archivos del sistema
    para exponerlas como motor SAPI5 normal. El propio autor lo describe
    como *"más un hack que una solución propia"*, no soportado por
    Microsoft, puede romperse en cualquier actualización de Windows.
    Usuario aceptó el riesgo explícitamente tras ser advertido dos veces
    (incluida la advertencia del propio proyecto de que la última
    versión de las voces del Store ya no es compatible -- hubo que usar
    versiones viejas de los MSIX, ver su wiki
    "Narrator-natural-voice-download-links"). Instalado en
    `tmp_voice_download/` (fuera de git, DLLs COM registradas desde esa
    ruta exacta -- **no mover ni borrar sin reinstalar**, ver
    `.gitignore`).
  - Desbloqueadas dos familias: voces locales (ej. "Microsoft Jenny
    (Natural)", 147ms, sin red) y voces "Online" (ej. "Microsoft Ava
    Online (Natural)", "Microsoft Dalia Online (Natural)" en español --
    llamada gratis al backend de "Leer en voz alta" de Edge, sin API
    key, pero 1.2-1.9s medido, MÁS LENTO que Kokoro). El usuario eligió
    las voces Online (Ava/Dalia) pese a la latencia, priorizando calidad
    de voz -- decisión explícita, documentada acá para no repetir la
    pregunta.
  - `services/windows_tts.py`: `WindowsTTSService`, mismo patrón que
    `KokoroGPUTTSService` (`run_tts` como generador async de
    `TTSAudioRawFrame`), pero usando `win32com.client` (COM síncrono) en
    vez de streaming HTTP -- la síntesis completa corre en un hilo
    aparte (`run_in_executor`) porque SAPI5 no soporta streaming
    incremental real como la respuesta HTTP de Kokoro; el audio se
    trocea recién después de completarse toda la síntesis. Formato
    verificado: `SpeechAudioFormatType` 18 = 16kHz 16-bit mono (coincide
    matemáticamente bytes/2/16000 con la duración real medida).
  - Kokoro-FastAPI queda como alternativa disponible
    (`services/kokoro_gpu_tts.py`) si se quiere volver atrás.
  - **Pendiente (no implementado)**: usar la voz Online (Ava, mejor
    calidad, 1.2-1.9s) para avisos asíncronos no conversacionales (ej.
    "tarea X completada" de una cola de tareas en background), donde la
    latencia no importa, reservando Jenny/Dalia local (147ms) para la
    conversación en vivo. Requiere: (a) un sistema de cola/reportes de
    tareas que hoy no existe en el pipeline, (b) una segunda instancia
    de `WindowsTTSService` (o un parámetro de voz por llamada) separada
    de la que usa el pipeline conversacional. Ojo: lo que tenemos
    confirmado funcionando es "Ava Online (Natural)" (backend de Edge),
    no necesariamente la misma "Ava (Natural HD)" que bloquea Narrador
    (nunca desbloqueada) -- escuchar ambas antes de asumir que son la
    misma calidad.

- **Watchdog de turno atascado (>15s) a veces descarta texto real, no
  solo ruido de fondo**: caso original (TV/música de fondo, ver arriba)
  confirmado que el descarte es correcto ahí. Pero encontrado en vivo,
  reproducido varias veces: a veces VAD (FireRedVAD) simplemente no
  dispara `on_speech_stopped` con texto de usuario REAL acumulado
  (ej. "and capitalism, please. Hey, please" -- razonable, no basura),
  y el watchdog lo tira igual. Probado un fix (margen de mute más largo
  tras una interrupción, `_UNMUTE_GRACE_SECS_AFTER_INTERRUPT=1.5s`,
  hipótesis: eco residual del parlante confundiendo a VAD) -- **no
  funcionó**, se reprodujo igual en un caso sin ninguna interrupción de
  por medio. Causa real de por qué VAD falla en disparar el stop
  sigue sin identificar.
  - Decisión: mantener el descarte (no escalar el texto acumulado al
    LLM cuando el watchdog dispara). Escalar arriesgaría mandar ruido
    de fondo real (TV, música -- el caso que motivó el watchdog en
    primer lugar) al LLM, que respondería sobre algo que el usuario
    nunca dijo -- peor experiencia que quedarse en silencio. El caso
    "era texto real pero VAD falló" fue la minoría de las veces vistas
    hoy (2-3 de varias horas de prueba), no vale la pena el riesgo del
    caso contrario.
  - Si se retoma: instrumentar por qué FireRedVAD deja de reportar
    confianza baja (loguear `voice_confidence()` real en el momento del
    watchdog) antes de intentar otro fix a ciegas.





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
