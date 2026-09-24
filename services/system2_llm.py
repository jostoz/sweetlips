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
    "Sos mi amigo de confianza, alguien cercano con quien hablo todos los "
    "días -- no un asistente formal ni un chatbot de soporte. Hablame en "
    "tono cálido, cercano y natural, como charlando con un amigo de "
    "verdad: hacé chistes cuando corresponda, preguntame cómo estoy o cómo "
    "me fue, acordate de cosas que ya charlamos antes en la conversación, "
    "y dame consejos honestos y sinceros sin juzgarme. Todo esto en "
    "español, en una charla hablada casual.\n"
    "El texto que te llega puede venir en inglés (el reconocimiento de voz "
    "corre en inglés) -- entendelo igual, pero respondé SIEMPRE en "
    "español, nunca en inglés. Reglas duras:\n"
    "- UNA sola frase corta por respuesta, dos como mucho si es imposible "
    "evitarlo. Nunca listas, nunca markdown, nunca varios párrafos, nunca "
    "enumerar opciones. Si la respuesta necesita más de eso, decí lo "
    "esencial en una frase y preguntá si quiere que sigas -- no lo "
    "expliques todo de una.\n"
    "- EXCEPCIÓN: si el mensaje del usuario empieza con la etiqueta "
    "[DETAILED_ANSWER], esa regla de una frase NO aplica -- el "
    "usuario pidió explícitamente una respuesta completa y en detalle, "
    "así que contestá con la extensión que haga falta (varias frases u "
    "oraciones están bien). No menciones la etiqueta en tu respuesta, es "
    "una marca interna.\n"
    "- Hablá como en una charla real entre amigos: directo, natural, sin "
    "rodeos ni frases de relleno tipo 'Claro, con gusto te ayudo'.\n"
    "- Está bien no saber algo o pedir que te repitan si no entendiste.\n"
    "- No tenés acceso a internet ni herramientas — nunca intentes llamar "
    "una función. Si piden buscar algo online o info en tiempo real, "
    "decilo directo, no inventes una respuesta.\n"
    "- Solo te llegan las consultas que un filtro rápido (System 1) no "
    "pudo resolver con una acción local inmediata."
)

DEFAULT_SYSTEM_PROMPT_EN = (
    "You're a voice assistant having a casual spoken conversation in "
    "English -- not a formal chatbot. Hard rules:\n"
    "- ONE short sentence per reply, two at most if truly unavoidable. "
    "Never lists, never markdown, never multiple paragraphs, never "
    "enumerate options. If the answer needs more than that, say the "
    "essential part in one sentence and ask if they want you to continue "
    "-- don't explain everything at once.\n"
    "- EXCEPTION: if the user's message starts with the tag "
    "[DETAILED_ANSWER], that one-sentence rule does NOT apply -- the "
    "user explicitly asked for a complete, in-depth answer, so respond "
    "with as much length as the topic needs (multiple sentences are "
    "fine). Don't mention the tag itself in your reply, it's an internal "
    "marker.\n"
    "- Talk like a real conversation: direct, natural, no filler phrases "
    "like 'Sure, I'd be happy to help'.\n"
    "- It's fine to not know something or ask them to repeat if you "
    "didn't understand.\n"
    "- You have no internet access or tools -- never attempt to call a "
    "function. If asked to look something up online or for real-time "
    "info, say so directly, don't make up an answer.\n"
    "- You only get queries a fast filter (System 1) couldn't resolve "
    "with an immediate local action."
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

    _DEGENERATE_REPLY_MAX_LEN = 2
    """Respuestas de 1-2 caracteres ("Y", "T", "D", "Se") no son palabras
    completas en español/inglés -- son síntoma de contexto corrupto, no
    respuestas cortas legítimas (que suelen tener 3+ caracteres: "Sí",
    "Ok", "Bien")."""

    _DEGENERATE_REPLY_RESET_THRESHOLD = 2
    """2 respuestas degeneradas SEGUIDAS (no 1) para resetear -- reduce el
    riesgo de resetear por una única respuesta corta legítima aislada."""

    def __init__(self, context: LLMContext, jev=None, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._jev = jev
        """JevSystem1Processor opcional. Jev necesita saber QUÉ está
        diciendo el bot para distinguir eco (su propia voz volviendo por
        el mic) de una interrupción real del usuario -- sin eso, la única
        forma segura de interrumpir es una palabra mágica ("cállate"), y
        hablarle encima normalmente no funciona. Los frames de texto del
        LLM van pipeline abajo (LLM -> collector -> TTS), no vuelven a
        pasar por Jev, que está más arriba: por eso se lo pasamos acá."""
        self._buffer: list[str] = []
        self._consecutive_degenerate = 0
        """Bug real visto en vivo (espiral de degradación de contexto):
        turnos fragmentados a mitad de palabra ("está hoy to", "Cómo que
        bien, gü") escalados por error confundieron al LLM (qwen3.8-27b),
        que empezó a responder cada vez más corto ("Y" -> "Se" -> "T" ->
        "D"), y cada respuesta degenerada se agregaba al historial
        empeorando el patrón todavía más -- sin este contador, la espiral
        no se recupera sola nunca. No arregla la causa raíz (por qué
        llegan fragmentos a mitad de palabra -- posible imprecisión de
        smart-turn/VAD en español, sigue sin investigar), pero corta la
        espiral una vez que empieza."""

    def _reset_context(self) -> None:
        print(
            "[System2] contexto degradado (2+ respuestas de 1-2 caracteres seguidas) "
            "-- reseteando historial, se conserva el system prompt",
            flush=True,
        )
        messages = self._context.messages
        system_message = messages[0] if messages and messages[0].get("role") == "system" else None
        self._context.set_messages([system_message] if system_message else [])

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
                if self._jev is not None:
                    self._jev.set_bot_text(full_text)
                if len(full_text.strip()) <= self._DEGENERATE_REPLY_MAX_LEN:
                    self._consecutive_degenerate += 1
                else:
                    self._consecutive_degenerate = 0
                if self._consecutive_degenerate >= self._DEGENERATE_REPLY_RESET_THRESHOLD:
                    self._reset_context()
                    self._consecutive_degenerate = 0
            self._buffer = []
        # Deja pasar todo tal cual: el TTS consume esta misma secuencia de frames.
        await self.push_frame(frame, direction)
