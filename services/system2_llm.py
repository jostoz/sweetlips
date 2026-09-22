"""System 2: puente de contexto hacia el LLM cuando Jev escala una consulta.

Dos processors livianos, sin usar el `LLMContextAggregatorPair` completo de
Pipecat (ese trae su propia gestión de turnos basada en VAD/STT que pisaría
la segmentación de turnos que ya hace Jev/System 1). En su lugar:

- `System2PromptBridge` (antes del LLM): recibe el `TextFrame` que Jev emite
  al escalar, lo agrega como mensaje "user" al `LLMContext` compartido, y
  dispara la inferencia empujando un `LLMContextFrame` río abajo.
- `System2ResponseCollector` (después del LLM, antes del TTS): escucha el
  streaming de la respuesta (`LLMFullResponseStartFrame` /
  `LLMTextFrame` / `LLMFullResponseEndFrame`) para guardar la respuesta del
  asistente en el mismo `LLMContext` (memoria multi-turno) y deja pasar los
  frames sin tocarlos — el TTS ya sabe consumir esa secuencia nativamente.
"""

from __future__ import annotations

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from services import latency_probe

DEFAULT_SYSTEM_PROMPT = (
    "Sos un asistente de voz conversando en español, en una charla hablada "
    "casual — no un chatbot formal. Reglas duras:\n"
    "- UNA sola frase corta por respuesta, dos como mucho si es imposible "
    "evitarlo. Nunca listas, nunca markdown, nunca varios párrafos, nunca "
    "enumerar opciones. Si la respuesta necesita más de eso, decí lo "
    "esencial en una frase y preguntá si quiere que sigas -- no lo "
    "expliques todo de una.\n"
    "- Hablá como en una charla real: directo, natural, sin rodeos ni "
    "frases de relleno tipo 'Claro, con gusto te ayudo'.\n"
    "- Está bien no saber algo o pedir que te repitan si no entendiste.\n"
    "- No tenés acceso a internet ni herramientas — nunca intentes llamar "
    "una función. Si piden buscar algo online o info en tiempo real, "
    "decilo directo, no inventes una respuesta.\n"
    "- Solo te llegan las consultas que un filtro rápido (System 1) no "
    "pudo resolver con una acción local inmediata."
)


def build_shared_context(system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> LLMContext:
    """Crea el `LLMContext` que comparten el bridge de prompt y el de respuesta."""
    return LLMContext(messages=[{"role": "system", "content": system_prompt}])


class System2PromptBridge(FrameProcessor):
    """Convierte el `TextFrame` escalado por Jev en un turno de LLM."""

    _FALLBACK_TEXT = "Perdón, no entendí bien. ¿Podés repetir?"
    """Groq/gpt-oss-120b tiene un bug conocido (reportado en vLLM, LangChain,
    HF) donde a veces el streaming de razonamiento rompe el parser de la
    respuesta y el turno se pierde por completo -- el ErrorFrame viaja
    río arriba (`push_error_frame`) y nunca llega al TTS, así que sin este
    fallback el usuario se queda en silencio total, indistinguible de que
    el sistema no le respondió nada."""

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, ErrorFrame) and direction == FrameDirection.UPSTREAM:
            print(f"[System2] LLM falló ({frame.error}) -> fallback hablado", flush=True)
            await self.push_frame(TextFrame(text=self._FALLBACK_TEXT), FrameDirection.DOWNSTREAM)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TextFrame) and not isinstance(frame, LLMTextFrame):
            self._context.add_message({"role": "user", "content": frame.text})
            latency_probe.mark("prompt enviado al LLM (Groq)")
            await self.push_frame(LLMContextFrame(context=self._context), direction)
            return

        await self.push_frame(frame, direction)


class System2ResponseCollector(FrameProcessor):
    """Guarda la respuesta del LLM en el contexto compartido (memoria multi-turno)."""

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._buffer: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = []
            latency_probe.mark("LLM: primer token")
        elif isinstance(frame, LLMTextFrame):
            self._buffer.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            full_text = "".join(self._buffer)
            latency_probe.mark("LLM: respuesta completa")
            if full_text:
                self._context.add_message({"role": "assistant", "content": full_text})
                print(f"[System2] respuesta del LLM: \"{full_text}\"", flush=True)
            self._buffer = []
        # Deja pasar todo tal cual: el TTS consume esta misma secuencia de frames.
        await self.push_frame(frame, direction)
