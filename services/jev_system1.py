"""System 1 (Jev): enrutador rápido de baja latencia sobre el texto append-only
que emite el ASR (Confucius4-R2T2).

Decide, por cada `TranscriptionFrame` confirmado:
  1. Interrupción inmediata (barge-in) por palabra clave.
  2. Acción local atómica sin pasar por LLM.
  3. Escalado a System 2 (LLM) para razonamiento/conversación (por palabra
     clave, o por defecto al terminar el turno si no matcheó nada más).
  4. Nada aún: la frase sigue abierta, se espera el siguiente token.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from services import latency_probe
from services import semantic_jev_router

from actions.local_dispatcher import execute_local_command

_INTERRUPT_WORDS_BY_LANG = {
    "English": ("stop", "shut up", "quiet", "cancel", "enough"),
    "Spanish": ("cállate", "callate", "detente", "silencio", "cancela"),
    # "para" (sola) estaba antes en español: es una de las palabras más
    # comunes ("cosas para picar", "bueno para comer"...) y sin
    # auriculares el bot se autointerrumpía al escucharse decir su propia
    # "para" por el parlante -> mic -> ASR. Sacada; las que quedan son
    # comandos explícitos de corte que casi nunca aparecen sueltos en una
    # frase normal.
}
_INTERRUPT_PREFIX_LEN = 4
# Match por prefijo, no la palabra completa: si el usuario escala el
# turno (o R2T2 tarda en transcribir) antes de terminar de decir
# "cállate", el texto queda cortado en algo como "cáll" -- eso NO
# contiene "cállate" como substring completo y el corte no se disparaba
# (bug real, visto en vivo: "pará, cállate" llegó como "para cáll" y se
# mandó al LLM en vez de interrumpir). Alcanza con los primeros 4
# caracteres de cada palabra para no confundirse con otras.


def _has_interrupt_word(normalized_text: str, language: str) -> bool:
    words = _INTERRUPT_WORDS_BY_LANG.get(language, _INTERRUPT_WORDS_BY_LANG["English"])
    return any(
        word in normalized_text or word[:_INTERRUPT_PREFIX_LEN] in normalized_text
        for word in words
    )

_ESCALATE_WORDS = ("por qué", "por que", "cómo", "como", "explícame", "explicame", "recomiéndame", "recomiendame")
# ^ Ya NO se usa para escalar a mitad de frase (ver nota abajo en
# _evaluate_intent) -- se deja documentado por si se reintroduce algo
# similar con mejores garantías (ej: sólo si el texto ya tiene >N
# palabras, o sólo al inicio del turno).

_SLOW_PATH_TAG = "[DETAILED_ANSWER]"
# Prefijo interno agregado al TextFrame para pedirle al LLM que ignore el
# límite de una frase (ver DEFAULT_SYSTEM_PROMPT_EN en system2_llm.py).

_SLOW_PATH_CHIME_SAMPLE_RATE = 24000


def _generate_chime(sample_rate: int) -> bytes:
    """Dos notas ascendentes cortas (señal de "pensando", no hablada --
    más rápida que un TTSSpeakFrame porque no pasa por síntesis: es
    audio crudo, suena en cuanto se empuja el frame). Envelope con
    fade in/out de 10ms en cada nota para evitar clicks."""
    notes_hz = (720.0, 1080.0)
    note_secs = 0.09
    fade_secs = 0.01
    chunks = []
    for freq in notes_hz:
        n = int(sample_rate * note_secs)
        t = np.arange(n) / sample_rate
        wave = np.sin(2 * np.pi * freq * t)
        fade_n = int(sample_rate * fade_secs)
        envelope = np.ones(n)
        envelope[:fade_n] = np.linspace(0.0, 1.0, fade_n)
        envelope[-fade_n:] = np.linspace(1.0, 0.0, fade_n)
        chunks.append(wave * envelope * 0.3)
    samples = np.concatenate(chunks)
    return (samples * 32767).astype(np.int16).tobytes()


_SLOW_PATH_CHIME_AUDIO = _generate_chime(_SLOW_PATH_CHIME_SAMPLE_RATE)

# La decisión fast/slow ya NO es keyword/largo-de-frase: la heurística de
# keywords no generalizaba (bug real en vivo: "Make analysis of the two
# principal ideologues, capitalism and socialism" no matcheaba ninguna
# keyword y cayó al fast path truncado). Ahora usa
# `semantic_jev_router.is_slow_path(texto, idioma)`, matching semántico
# local (FastEmbedEncoder, sin API/red) que generaliza a paráfrasis nunca
# vistas, parametrizado por idioma (ver services/semantic_jev_router.py).
# Se llama una sola vez por turno en _escalate() (no por delta): ~30-60ms
# medidos, insignificante contra el presupuesto de ~400-500ms del turno
# completo -- correrlo por delta sí sería costoso.


class JevSystem1Processor(FrameProcessor):
    """Router System 1: interrumpe, resuelve localmente o escala a System 2."""

    _UNMUTE_GRACE_SECS = 0.6
    """Tiempo extra silenciado tras `BotStoppedSpeakingFrame`: el audio
    físico sigue sonando por el parlante un rato después de que pipecat
    considera que el bot "terminó" (buffer de reproducción), y sin eso el
    mic se re-transcribe a sí mismo justo en ese hueco."""

    _TURN_END_DEBOUNCE_SECS = 0.5
    """Al detectar VADUserStoppedSpeakingFrame (0.7s de silencio de VAD) no
    escalamos al instante: es común hacer una pausa corta para pensar y
    seguir la misma idea 1-2s después ("pláticame qué lugares conoces
    tú... pláticame qué lugares conoc[es]" se cortaba justo ahí). Con
    este debounce, si el usuario retoma antes de que se cumpla el tiempo
    (VADUserStartedSpeakingFrame), cancelamos la escalada pendiente y
    seguimos acumulando texto en el mismo turno en vez de perderlo.
    Vuelto a 0.5s (de 0.35s): el patrón real en vivo fue perder
    sistemáticamente la palabra clave justo tras una preposición
    ("historia de", "ciudad de" -- pausa pensando, corte antes de tiempo).
    Precisión > latencia."""

    def __init__(
        self,
        r2t2_stt=None,
        smart_turn: BaseTurnAnalyzer | None = None,
        language: str = "English",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._language = language
        """Mismo string que ConfuciusR2T2Service.language ("English"/
        "Spanish"): elige el set de palabras de interrupción/acciones
        locales y el encoder+rutas del router semántico (ver
        services/semantic_jev_router.py)."""
        self.confirmed_text = ""
        self._bot_speaking = False
        self._unmute_task: asyncio.Task | None = None
        self._pending_escalate_task: asyncio.Task | None = None
        self._r2t2_stt = r2t2_stt
        """Referencia opcional a ConfuciusR2T2Service: si está seteada, se
        le pide flush_final() antes de escalar (ver _debounced_turn_end)
        para no perder la cola de la última palabra dicha."""
        self._mute_watch_text = ""
        """Buffer separado de confirmed_text, usado SOLO para detectar
        interrupción mientras el bot habla. confirmed_text se resetea
        por delta durante el mute (para no filtrar eco a un turno real
        después) -- pero eso rompía la detección de interrupción cuando
        R2T2 parte la palabra en varios deltas ("cállate" -> "Cá"+"ll"+
        "ate"): cada fragmento se revisaba aislado y nunca matcheaba.
        Este buffer sí acumula entre deltas (acotado a los últimos 40
        caracteres) para poder detectar la palabra completa aunque
        llegue partida, sin arriesgar que texto de eco se filtre al
        turno real (nunca se usa para escalar, solo para este chequeo)."""
        self._smart_turn = smart_turn
        """Analizador semántico de fin de turno (modelo ONNX, ver
        pipecat.audio.turn.smart_turn) opcional. Si está seteado,
        reemplaza el debounce por timer fijo (_TURN_END_DEBOUNCE_SECS)
        con una decisión real: ¿el audio suena a que el usuario terminó
        de hablar, o está a mitad de una idea? Evita el trade-off
        latencia/precisión de los timers fijos (subirlos = más preciso
        pero más lento SIEMPRE, incluso cuando el usuario sí terminó)."""
        self._user_speaking = False
        self._sample_rate = 16000
        self._turn_started_at: float | None = None
        """Timestamp (time.monotonic()) del primer VADUserStartedSpeakingFrame
        de un turno todavía sin resolver. None cuando no hay turno abierto.
        Ver _MAX_TURN_DURATION_SECS: sin esto, ruido de fondo continuo (TV,
        música -- sin pausas de silencio reales) deja el turno abierto para
        siempre, porque VAD nunca dispara VADUserStoppedSpeakingFrame y por
        lo tanto ni el debounce ni smart-turn se llegan a evaluar. Bug real
        confirmado en vivo: turno de más de un minuto acumulando texto sin
        resolver con la TV de fondo."""
        self._watchdog_task: asyncio.Task | None = None
        self._grace_period_pending = False
        self._grace_period_saves = 0
        self._grace_period_wastes = 0
        """Medición temporal (sesión de hoy) para decidir si
        _SMART_TURN_COMPLETE_GRACE_SECS sigue valiendo la pena en inglés
        -- cuenta cuántas veces el grace period evita un corte real
        (usuario retoma, "saves") vs cuántas veces solo agrega espera sin
        que hiciera falta (usuario no retoma, "wastes")."""

    async def setup(self, setup) -> None:
        await super().setup(setup)
        if self._smart_turn is not None:
            self._smart_turn.set_sample_rate(self._sample_rate)
        # Precarga el encoder ONNX del router semántico (~1.5-2s la primera
        # vez, descarga+carga del modelo) acá, no en el primer turno real --
        # sin esto, la primera vez que alguien escala a slow path esperaría
        # ese delay entero antes de que suene el chime. run_in_executor:
        # es una llamada bloqueante (CPU-bound, carga de modelo), no async.
        await asyncio.get_event_loop().run_in_executor(
            None, semantic_jev_router.warm_up, self._language
        )
        self._watchdog_task = asyncio.create_task(self._max_turn_watchdog())
        # Bug real visto en vivo: el chequeo de _MAX_TURN_DURATION_SECS vivía
        # antes DENTRO del handler de InputAudioRawFrame -- si el transporte
        # de audio se congela por completo (sin excepción, sin log, mic
        # simplemente deja de entregar frames), ese chequeo nunca se
        # disparaba porque depende de que sigan llegando frames nuevos para
        # evaluarse. Turno quedó "escuchado: Hola, cómo te" colgado 7+
        # minutos sin ningún error. Watchdog independiente (loop propio,
        # no depende de que fluya audio) sí lo detecta pase lo que pase río
        # arriba.

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            # El bot va a hablar: silenciar el ASR para no re-transcribir su
            # propia voz por el micrófono (sin auriculares hay acople).
            if self._unmute_task is not None:
                self._unmute_task.cancel()
                self._unmute_task = None
            if self._pending_escalate_task is not None:
                self._pending_escalate_task.cancel()
                self._pending_escalate_task = None
            latency_probe.mark("bot empieza a hablar (audio real)")
            self._bot_speaking = True
            # (Ya NO se resetea confirmed_text acá.) `_escalate()` ya lo
            # vació al armar el turno del LLM, mucho antes de que el bot
            # empiece a hablar -- resetear de nuevo acá era redundante Y
            # peligroso: si la respuesta del bot tiene varias oraciones
            # (TTS separado por frase) y el usuario ya empezó a hablar de
            # nuevo antes de que termine de sonar la última, un
            # BotStoppedSpeakingFrame tardío podía borrar lo que el
            # usuario ya llevaba dicho ("Contame algo sobre el espacio"
            # se perdía todo salvo "espacio").
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            if self._unmute_task is not None:
                self._unmute_task.cancel()
            self._unmute_task = asyncio.create_task(self._unmute_after_grace())
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame, direction)
            return

        if isinstance(frame, InputAudioRawFrame):
            if self._smart_turn is not None:
                self._smart_turn.append_audio(frame.audio, self._user_speaking)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # El usuario retomó antes de que se cumpliera el debounce/
            # análisis de fin de turno: cancelar lo pendiente, seguir
            # acumulando en el mismo turno (NO resetear confirmed_text).
            self._user_speaking = True
            if self._turn_started_at is None:
                self._turn_started_at = time.monotonic()
            if self._pending_escalate_task is not None:
                self._pending_escalate_task.cancel()
                self._pending_escalate_task = None
                if self._grace_period_pending:
                    self._grace_period_saves += 1
                    print(
                        f"[Jev] grace period SALVÓ un corte (usuario retomó) -- "
                        f"{self._grace_period_saves} salvados / {self._grace_period_saves + self._grace_period_wastes} totales",
                        flush=True,
                    )
                    self._grace_period_pending = False
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            await self.push_frame(frame, direction)
            if self._pending_escalate_task is not None:
                self._pending_escalate_task.cancel()
            if self._smart_turn is not None:
                self._pending_escalate_task = asyncio.create_task(self._smart_turn_end(direction))
            else:
                self._pending_escalate_task = asyncio.create_task(self._debounced_turn_end(direction))
            return

        await self.push_frame(frame, direction)

    _MAX_TURN_DURATION_SECS = 15.0
    """Tope duro: si un turno lleva abierto más de esto sin que VAD dispare
    VADUserStoppedSpeakingFrame, se descarta a la fuerza (ver
    _force_reset_stuck_turn). Chequeado por un watchdog independiente
    (_max_turn_watchdog, loop propio con su timer, NO depende de que
    lleguen frames de audio) -- bug real: un chequeo anterior vivía dentro
    del handler de InputAudioRawFrame y nunca se disparaba si el
    transporte de audio se congelaba por completo (turno quedó colgado
    7+ minutos sin ningún error ni log). También cubre el caso original:
    TV/música de fondo sin pausas de silencio deja el turno abierto
    indefinidamente -- el AEC (WASAPI loopback) sólo cancela el eco de lo
    que el PROPIO sistema reproduce, no audio ambiente de una fuente
    separada."""

    _MAX_TURN_WATCHDOG_INTERVAL_SECS = 5.0
    """Cada cuánto revisa el watchdog si hay un turno atascado. No hace
    falta más fino que esto -- el tope real es _MAX_TURN_DURATION_SECS."""

    async def _max_turn_watchdog(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._MAX_TURN_WATCHDOG_INTERVAL_SECS)
                if (
                    self._turn_started_at is not None
                    and time.monotonic() - self._turn_started_at > self._MAX_TURN_DURATION_SECS
                ):
                    await self._force_reset_stuck_turn()
        except asyncio.CancelledError:
            pass

    async def _force_reset_stuck_turn(self) -> None:
        print(
            f'[Jev] turno atascado >{self._MAX_TURN_DURATION_SECS:.0f}s sin silencio '
            f'(¿ruido de fondo o transporte de audio congelado?) -- descartando: '
            f'"{self.confirmed_text.strip()[:80]}..."',
            flush=True,
        )
        self.confirmed_text = ""
        self._mute_watch_text = ""
        self._turn_started_at = None
        if self._pending_escalate_task is not None:
            self._pending_escalate_task.cancel()
            self._pending_escalate_task = None
        if self._r2t2_stt is not None:
            # Fuerza a R2T2 a cerrar y reabrir la conexión (ver flush_final):
            # sin esto, el turno "atascado" en el servidor seguiría
            # acumulando audio/texto viejo para la próxima vez que sí haya
            # silencio real.
            await self._r2t2_stt.flush_final()


    _SMART_TURN_INCOMPLETE_FALLBACK_SECS = 2.5
    """Si el modelo de smart-turn dice INCOMPLETE (cree que el usuario va a
    seguir hablando), no escalamos todavía -- pero si no retoma en este
    tiempo, escalamos igual. Red de seguridad: el modelo puede
    equivocarse, y no queremos dejar al usuario esperando para siempre."""

    _SMART_TURN_COMPLETE_GRACE_SECS = 0.35
    """Aunque smart-turn diga COMPLETE, no escalar al instante -- esperar
    este margen por si el usuario retoma. Bug real visto en vivo:
    "Hola, ¿cómo estás? Podrías" escaló completo y cortado a mitad de
    frase (confirmado por timing: pasó ANTES de que el bot empezara a
    hablar, no fue el mute comiéndose el inicio -- smart-turn-v3.2
    (última versión, con soporte de español) igual se equivocó en una
    pausa natural para pensar). Reusa _debounced_turn_end, que ya se
    cancela solo si llega VADUserStartedSpeakingFrame en la ventana
    (mismo mecanismo que ya existía para el camino INCOMPLETE)."""

    async def _smart_turn_end(self, direction: FrameDirection) -> None:
        state, _ = await self._smart_turn.analyze_end_of_turn()
        self._pending_escalate_task = None
        if state == EndOfTurnState.COMPLETE:
            self._grace_period_pending = True
            self._pending_escalate_task = asyncio.create_task(
                self._debounced_turn_end(direction, delay=self._SMART_TURN_COMPLETE_GRACE_SECS)
            )
        else:
            print("[Jev] smart-turn: incompleto, espero que el usuario siga", flush=True)
            self._pending_escalate_task = asyncio.create_task(
                self._debounced_turn_end(direction, delay=self._SMART_TURN_INCOMPLETE_FALLBACK_SECS)
            )

    async def _debounced_turn_end(self, direction: FrameDirection, delay: float | None = None) -> None:
        was_grace_period = self._grace_period_pending
        try:
            await asyncio.sleep(delay if delay is not None else self._TURN_END_DEBOUNCE_SECS)
        except asyncio.CancelledError:
            return
        self._pending_escalate_task = None
        if was_grace_period:
            self._grace_period_wastes += 1
            self._grace_period_pending = False
            print(
                f"[Jev] grace period no hizo falta (usuario no retomó) -- "
                f"{self._grace_period_saves} salvados / {self._grace_period_saves + self._grace_period_wastes} totales",
                flush=True,
            )
        if self._r2t2_stt is not None:
            tail = await self._r2t2_stt.flush_final()
            if tail:
                self.confirmed_text += tail
                print(f'[Jev] flush R2T2 recuperó cola: "{tail}"', flush=True)
        await self._handle_turn_end(direction)

    async def _unmute_after_grace(self) -> None:
        try:
            await asyncio.sleep(self._UNMUTE_GRACE_SECS)
            self._bot_speaking = False
            # (Ya NO se resetea confirmed_text acá, mismo motivo que en
            # BotStoppedSpeakingFrame: el reset por delta dentro del
            # branch mute de _handle_transcription ya cubre el caso
            # legítimo -- resetear acá de más podía borrar texto real que
            # el usuario ya empezó a decir apenas terminó el grace.)
        except asyncio.CancelledError:
            pass

    async def _handle_transcription(self, frame: TranscriptionFrame, direction: FrameDirection) -> None:
        # R2T2 emite deltas append-only: el espaciado entre palabras ya viene
        # correcto en el propio delta. Concatenar directo, sin insertar
        # espacios ni recortar bordes (eso rompía palabras: "esc uch as").
        if not frame.text:
            return

        self.confirmed_text += frame.text
        print(f'[Jev] escuchado: "{self.confirmed_text.strip()}"', flush=True)

        normalized = self.confirmed_text.strip().lower()

        if self._bot_speaking:
            # Se probó relajar esto (confiar en el AEC) y el AEC no
            # cancela lo suficiente: el bot volvió a escucharse a sí
            # mismo ("se está escuchando otra vez", confirmado en vivo).
            # Vuelta a la regla segura: mientras el bot habla, ignoramos
            # todo salvo un barge-in explícito.
            #
            # confirmed_text se resetea por delta (no filtrar eco a un
            # turno real), pero eso rompía la detección de interrupción
            # cuando R2T2 parte la palabra en varios deltas ("cállate"
            # -> "Cá"+"ll"+"ate": cada fragmento revisado aislado nunca
            # matcheaba con confirmed_text solo). _mute_watch_text sí
            # acumula entre deltas (acotado) para cubrir ese caso.
            self._mute_watch_text = (self._mute_watch_text + frame.text)[-40:]
            if _has_interrupt_word(self._mute_watch_text.lower(), self._language):
                print("[Jev] -> interrupción detectada, cortando TTS", flush=True)
                await self.broadcast_interruption()
                self._mute_watch_text = ""
            self.confirmed_text = ""
            return

        # 1. Reflejo de interrupción (barge-in): corta System 2/TTS al instante.
        if _has_interrupt_word(normalized, self._language):
            print("[Jev] -> interrupción detectada, cortando TTS", flush=True)
            await self.broadcast_interruption()
            self.confirmed_text = ""
            self._turn_started_at = None
            return


        decision = self._evaluate_intent(normalized)

        if decision["type"] == "LOCAL_ACTION":
            latency_probe.mark_turn_start()
            # Sin esto, mark("bot empieza a hablar") usa el _t0 de la
            # ÚLTIMA escalada a LLM (mark_turn_start solo se llamaba en
            # _escalate) -- bug real visto en vivo: una acción local
            # ("What time is it") reportó [LAT] t=11622ms porque el
            # timer venía de un turno de slow-path varios segundos
            # antes. La latencia real era ~170ms. También contaminaba
            # el histograma de Prometheus con outliers falsos.
            result_speech = execute_local_command(decision["action"], decision["target"], self._language)
            print(f'[Jev] -> acción local: {decision["action"]} {decision["target"]} => "{result_speech}"', flush=True)
            self.confirmed_text = ""
            self._turn_started_at = None
            # TTSSpeakFrame (no TextFrame): habla directo sin pasar por
            # System2PromptBridge. Bug real encontrado en vivo: con
            # TextFrame, System2PromptBridge lo interceptaba como si
            # fuera texto del USUARIO (mismo chequeo que usa para el
            # texto escalado por _escalate) y lo re-enviaba al LLM como
            # un mensaje nuevo -- "Luces encendidas." terminaba
            # preguntándole al LLM "¿por qué dijiste eso?", generando
            # respuestas sin sentido y latencia extra en cascada.
            await self.push_frame(TTSSpeakFrame(text=result_speech), direction)
            return

        if decision["type"] == "ESCALATE_SYSTEM_2":
            await self._escalate(direction)
            return

        # PENDING: la frase sigue abierta; esperamos el siguiente token de R2T2
        # o el fin del turno (VADUserStoppedSpeakingFrame) para decidir.

    async def _handle_turn_end(self, direction: FrameDirection) -> None:
        """El usuario dejó de hablar. Si quedó texto sin resolver, escalar
        a System 2 por defecto (no todo pasa por palabras clave)."""
        if self.confirmed_text.strip():
            await self._escalate(direction)

    async def _escalate(self, direction: FrameDirection) -> None:
        prompt = self.confirmed_text.strip()
        latency_probe.mark_turn_start()
        self.confirmed_text = ""
        self._turn_started_at = None
        if semantic_jev_router.is_slow_path(prompt, self._language):
            # Slow path (patrón FXPerto QueryRouter): chime no hablado
            # (dos notas ascendentes, audio crudo) + TextFrame con la
            # etiqueta que le saca a Groq el límite de una frase. Audio
            # crudo en vez de TTSSpeakFrame: suena en cuanto se empuja
            # el frame, sin esperar síntesis, y como no es
            # TTSAudioRawFrame no dispara BotStartedSpeakingFrame (no
            # hace falta silenciar el ASR por un chime de 180ms). El LLM
            # arma la respuesta larga en paralelo (frames de pipecat ya
            # son async, no hace falta asyncio.create_task acá).
            print(f'[Jev] -> escalando a System 2 (LLM, slow path): "{prompt}"', flush=True)
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=_SLOW_PATH_CHIME_AUDIO,
                    sample_rate=_SLOW_PATH_CHIME_SAMPLE_RATE,
                    num_channels=1,
                ),
                direction,
            )
            prompt = f"{_SLOW_PATH_TAG} {prompt}"
        else:
            print(f'[Jev] -> escalando a System 2 (LLM): "{prompt}"', flush=True)
        await self.push_frame(TextFrame(text=prompt), direction)

    def _evaluate_intent(self, text: str) -> dict:
        """Reglas rápidas de Jev (System 1). Objetivo: decidir en <10ms."""
        words = text.split()

        if self._language == "Spanish":
            if "enciende" in text and "luz" in text:
                return {"type": "LOCAL_ACTION", "action": "TURN_ON", "target": "LIGHTS"}
            if "apaga" in text and "luz" in text:
                return {"type": "LOCAL_ACTION", "action": "TURN_OFF", "target": "LIGHTS"}
            # Word-list, no substring: "hora" como substring matchea "ahora"
            # ("¿y ahora qué hacemos?"), que no tiene nada que ver con pedir
            # la hora. El LLM contesta esto MAL ("no tengo acceso al reloj
            # en tiempo real") cuando la máquina sí sabe la hora -- resuelto
            # acá, sin ida y vuelta al LLM, determinístico.
            if "hora" in words or "horas" in words:
                return {"type": "LOCAL_ACTION", "action": "QUERY", "target": "TIME"}
            if "fecha" in words or ("qué" in words and "día" in words and "es" in words):
                return {"type": "LOCAL_ACTION", "action": "QUERY", "target": "DATE"}
        else:
            if "turn on" in text and "light" in text:
                return {"type": "LOCAL_ACTION", "action": "TURN_ON", "target": "LIGHTS"}
            if "turn off" in text and "light" in text:
                return {"type": "LOCAL_ACTION", "action": "TURN_OFF", "target": "LIGHTS"}
            # Word-list, no substring: avoids matching "time" inside e.g.
            # "sometimes". The LLM answers this WRONG ("I don't have access
            # to real-time clock data") when the machine actually knows the
            # time -- resolved here, no LLM round-trip, deterministic.
            if "time" in words:
                return {"type": "LOCAL_ACTION", "action": "QUERY", "target": "TIME"}
            if "date" in words:
                return {"type": "LOCAL_ACTION", "action": "QUERY", "target": "DATE"}


        # Antes escalaba a System 2 apenas aparecía una palabra tipo "cómo"
        # en el texto acumulado, SIN esperar a que el usuario terminara de
        # hablar -- eso disparaba el LLM con fragmentos truncados a mitad
        # de oración ("día, cómo" en vez de la pregunta completa), y el
        # resto de la transcripción que seguía llegando se perdía. La
        # escalada por defecto en _handle_turn_end (fin de turno real, vía
        # VADUserStoppedSpeakingFrame) ya cubre esto sin ese riesgo.

        return {"type": "PENDING"}
