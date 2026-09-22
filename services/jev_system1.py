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

from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from actions.local_dispatcher import execute_local_command

_INTERRUPT_WORDS = ("para", "cállate", "callate", "alto", "cancela")
_ESCALATE_WORDS = ("por qué", "por que", "cómo", "como", "explícame", "explicame", "recomiéndame", "recomiendame")


class JevSystem1Processor(FrameProcessor):
    """Router System 1: interrumpe, resuelve localmente o escala a System 2."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.confirmed_text = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame, direction)
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_turn_end(direction)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _handle_transcription(self, frame: TranscriptionFrame, direction: FrameDirection) -> None:
        # R2T2 emite deltas append-only: el espaciado entre palabras ya viene
        # correcto en el propio delta. Concatenar directo, sin insertar
        # espacios ni recortar bordes (eso rompía palabras: "esc uch as").
        if not frame.text:
            return

        self.confirmed_text += frame.text
        print(f'[Jev] escuchado: "{self.confirmed_text.strip()}"', flush=True)

        normalized = self.confirmed_text.strip().lower()

        # 1. Reflejo de interrupción (barge-in): corta System 2/TTS al instante.
        if any(word in normalized for word in _INTERRUPT_WORDS):
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
        print(f'[Jev] -> escalando a System 2 (LLM): "{prompt}"', flush=True)
        self.confirmed_text = ""
        await self.push_frame(TextFrame(text=prompt), direction)

    def _evaluate_intent(self, text: str) -> dict:
        """Reglas rápidas de Jev (System 1). Objetivo: decidir en <10ms."""
        if "enciende" in text and "luz" in text:
            return {"type": "LOCAL_ACTION", "action": "TURN_ON", "target": "LIGHTS"}
        if "apaga" in text and "luz" in text:
            return {"type": "LOCAL_ACTION", "action": "TURN_OFF", "target": "LIGHTS"}

        if any(word in text for word in _ESCALATE_WORDS):
            return {"type": "ESCALATE_SYSTEM_2"}

        return {"type": "PENDING"}
