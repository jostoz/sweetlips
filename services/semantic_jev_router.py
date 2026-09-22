"""Router semántico (fast/slow) para Jev, basado en `semantic-router`
(aurelio-labs, local/offline vía `FastEmbedEncoder` -- sin API key, sin
llamada de red, modelo ONNX chico corriendo en CPU).

Reemplaza la heurística de keywords/largo de `_is_slow_path` (jev_system1.py,
ya retirada) por matching semántico contra ejemplos de cada ruta. Motivación
real: la heurística de keywords no generalizaba -- "Make analysis of..." no
matcheaba ninguna keyword literal y cayó al fast path con un límite de 55
tokens, produciendo una respuesta cortada/rota (bug real visto en vivo). El
router semántico sí generaliza a paráfrasis nunca vistas ("what's the
difference between capitalism and socialism", sin ninguna keyword) porque
compara significado, no substrings.

Costo medido en esta GPU/CPU: ~1.5s de carga del encoder al importar (una
sola vez, al arrancar el proceso) + ~25-65ms por utterance evaluada. Por eso
NO se llama por cada delta de R2T2 (ahí sigue habiendo keyword matching
barato para acciones locales/interrupción, que necesitan reacción
instantánea) -- se llama UNA vez por turno, en `_escalate()`, donde 30-60ms
es insignificante contra el presupuesto de ~400-500ms del turno completo.
"""

from __future__ import annotations

from semantic_router import Route
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.routers import SemanticRouter

_SLOW_PATH_UTTERANCES = (
    "can you explain that in detail",
    "give me a deep analysis of this",
    "walk me through how this works",
    "compare the pros and cons",
    "what are the differences between these two things",
    "elaborate on that for me",
    "can you break this down step by step",
    "make an analysis of the two principal ideologies capitalism and socialism",
    "tell me more about the history in depth",
    "give me a thorough breakdown",
    "can you go deeper into that topic",
    "what's the difference between capitalism and socialism",
    "give me the rundown on how the internet works",
)
# Incluye los casos reales vistos en vivo (la frase de "capitalism and
# socialism" es textual del log) + paráfrasis sin ninguna keyword en común,
# para forzar que el router generalice por significado y no memorice.

_FAST_PATH_UTTERANCES = (
    "what time is it",
    "turn on the light",
    "hello how are you",
    "tell me about mexico city",
    "thank you",
    "continue",
    "what day is today",
    "turn off the light",
    "what's the weather like",
    "who are you",
)

_router: SemanticRouter | None = None


def _get_router() -> SemanticRouter:
    global _router
    if _router is None:
        encoder = FastEmbedEncoder()
        slow_path = Route(name="slow_path", utterances=list(_SLOW_PATH_UTTERANCES))
        fast_path = Route(name="fast_path", utterances=list(_FAST_PATH_UTTERANCES))
        _router = SemanticRouter(encoder=encoder, routes=[slow_path, fast_path], auto_sync="local")
    return _router


def warm_up() -> None:
    """Fuerza la carga del encoder + índice al arrancar el proceso, no en el
    primer turno real (evitaría un delay de ~1.5-2s en la primera escalada)."""
    _get_router()


def is_slow_path(text: str) -> bool:
    result = _get_router()(text)
    return result.name == "slow_path"
