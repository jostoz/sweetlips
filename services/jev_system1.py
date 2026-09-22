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

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    TextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from services import latency_probe

from actions.local_dispatcher import execute_local_command

_INTERRUPT_WORDS = ("cállate", "callate", "detente", "silencio", "cancela")
# "para" (sola) estaba antes: es una de las palabras más comunes del
# español ("cosas para picar", "bueno para comer"...) y sin auriculares
# el bot se autointerrumpía al escucharse decir su propia "para" por el
# parlante -> mic -> ASR. Sacada; las que quedan son comandos explícitos
# de corte que casi nunca aparecen sueltos en una frase normal.
_INTERRUPT_PREFIX_LEN = 4
# Match por prefijo, no la palabra completa: si el usuario escala el
# turno (o R2T2 tarda en transcribir) antes de terminar de decir
# "cállate", el texto queda cortado en algo como "cáll" -- eso NO
# contiene "cállate" como substring completo y el corte no se disparaba
# (bug real, visto en vivo: "pará, cállate" llegó como "para cáll" y se
# mandó al LLM en vez de interrumpir). Alcanza con los primeros 4
# caracteres de cada palabra para no confundirse con otras.


def _has_interrupt_word(normalized_text: str) -> bool:
    return any(
        word in normalized_text or word[:_INTERRUPT_PREFIX_LEN] in normalized_text
        for word in _INTERRUPT_WORDS
    )

_ESCALATE_WORDS = ("por qué", "por que", "cómo", "como", "explícame", "explicame", "recomiéndame", "recomiendame")
# ^ Ya NO se usa para escalar a mitad de frase (ver nota abajo en
# _evaluate_intent) -- se deja documentado por si se reintroduce algo
# similar con mejores garantías (ej: sólo si el texto ya tiene >N
# palabras, o sólo al inicio del turno).


class JevSystem1Processor(FrameProcessor):
    """Router System 1: interrumpe, resuelve localmente o escala a System 2."""

    _UNMUTE_GRACE_SECS = 0.6
    """Tiempo extra silenciado tras `BotStoppedSpeakingFrame`: el audio
    físico sigue sonando por el parlante un rato después de que pipecat
    considera que el bot "terminó" (buffer de reproducción), y sin eso el
    mic se re-transcribe a sí mismo justo en ese hueco."""

    _TURN_END_DEBOUNCE_SECS = 0.35
    """Al detectar VADUserStoppedSpeakingFrame (0.5s de silencio de VAD) no
    escalamos al instante: es común hacer una pausa corta para pensar y
    seguir la misma idea 1-2s después ("pláticame qué lugares conoces
    tú... pláticame qué lugares conoc[es]" se cortaba justo ahí). Con
    este debounce, si el usuario retoma antes de que se cumpla el tiempo
    (VADUserStartedSpeakingFrame), cancelamos la escalada pendiente y
    seguimos acumulando texto en el mismo turno en vez de perderlo.
    Bajado de 0.5s a 0.35s: sumado a los 0.5s de VAD, 0.7+0.5s totales se
    sentía lento para una charla conversacional."""

    def __init__(self, r2t2_stt=None, **kwargs):
        super().__init__(**kwargs)
        self.confirmed_text = ""
        self._bot_speaking = False
        self._unmute_task: asyncio.Task | None = None
        self._pending_escalate_task: asyncio.Task | None = None
        self._r2t2_stt = r2t2_stt
        """Referencia opcional a ConfuciusR2T2Service: si está seteada, se
        le pide flush_final() antes de escalar (ver _debounced_turn_end)
        para no perder la cola de la última palabra dicha."""

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

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # El usuario retomó antes de que se cumpliera el debounce de
            # fin de turno: cancelar la escalada pendiente, seguir
            # acumulando en el mismo turno (NO resetear confirmed_text).
            if self._pending_escalate_task is not None:
                self._pending_escalate_task.cancel()
                self._pending_escalate_task = None
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.push_frame(frame, direction)
            if self._pending_escalate_task is not None:
                self._pending_escalate_task.cancel()
            self._pending_escalate_task = asyncio.create_task(self._debounced_turn_end(direction))
            return

        await self.push_frame(frame, direction)

    async def _debounced_turn_end(self, direction: FrameDirection) -> None:
        try:
            await asyncio.sleep(self._TURN_END_DEBOUNCE_SECS)
        except asyncio.CancelledError:
            return
        self._pending_escalate_task = None
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
            if _has_interrupt_word(normalized):
                print("[Jev] -> interrupción detectada, cortando TTS", flush=True)
                await self.broadcast_interruption()
            self.confirmed_text = ""
            return

        # 1. Reflejo de interrupción (barge-in): corta System 2/TTS al instante.
        if _has_interrupt_word(normalized):
            print("[Jev] -> interrupción detectada, cortando TTS", flush=True)
            await self.broadcast_interruption()
            self.confirmed_text = ""
            return


        decision = self._evaluate_intent(normalized)

        if decision["type"] == "LOCAL_ACTION":
            result_speech = execute_local_command(decision["action"], decision["target"])
            print(f'[Jev] -> acción local: {decision["action"]} {decision["target"]} => "{result_speech}"', flush=True)
            self.confirmed_text = ""
            # Respuesta directa al TTS, sin pasar por el LLM (System 2).
            await self.push_frame(TextFrame(text=result_speech), direction)
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
        print(f'[Jev] -> escalando a System 2 (LLM): "{prompt}"', flush=True)
        self.confirmed_text = ""
        await self.push_frame(TextFrame(text=prompt), direction)

    def _evaluate_intent(self, text: str) -> dict:
        """Reglas rápidas de Jev (System 1). Objetivo: decidir en <10ms."""
        if "enciende" in text and "luz" in text:
            return {"type": "LOCAL_ACTION", "action": "TURN_ON", "target": "LIGHTS"}
        if "apaga" in text and "luz" in text:
            return {"type": "LOCAL_ACTION", "action": "TURN_OFF", "target": "LIGHTS"}

        # Antes escalaba a System 2 apenas aparecía una palabra tipo "cómo"
        # en el texto acumulado, SIN esperar a que el usuario terminara de
        # hablar -- eso disparaba el LLM con fragmentos truncados a mitad
        # de oración ("día, cómo" en vez de la pregunta completa), y el
        # resto de la transcripción que seguía llegando se perdía. La
        # escalada por defecto en _handle_turn_end (fin de turno real, vía
        # VADUserStoppedSpeakingFrame) ya cubre esto sin ese riesgo.

        return {"type": "PENDING"}
