"""Router semántico (fast/slow) para Jev, basado en `semantic-router`
(aurelio-labs, local/offline vía `FastEmbedEncoder` -- sin API key, sin
llamada de red, modelo ONNX chico corriendo en CPU).

Reemplaza la heurística de keywords/largo de `_is_slow_path` (jev_system1.py,
ya retirada) por matching semántico contra ejemplos de cada ruta. Motivación
real: la heurística de keywords no generalizaba (bug real en vivo: "Make
analysis of..." no matcheaba ninguna keyword literal y cayó al fast path con
un límite de 55 tokens, produciendo una respuesta cortada/rota). El router
semántico sí generaliza a paráfrasis nunca vistas ("what's the difference
between capitalism and socialism", sin ninguna keyword) porque compara
significado, no substrings.

Parametrizado por idioma (`language`, mismo string que usa
`ConfuciusR2T2Service`: "English"/"Spanish"): el encoder de inglés
(BAAI/bge-small-en-v1.5) NO sirve para español -- es un modelo mono-idioma,
entrenado solo en inglés. Para español se usa un encoder multilingüe
(paraphrase-multilingual-MiniLM-L12-v2), que además necesita un
score_threshold más bajo (0.3 vs 0.5 default): calibrado empíricamente
probando casos reales -- con el threshold default (0.5) el router devolvía
`None` (sin ruta) para preguntas claramente slow_path como "explicame paso a
paso cómo aprenden las redes neuronales".

Costo medido en esta GPU/CPU: ~1.5-2s de carga del encoder al importar (una
sola vez por idioma, cacheado) + ~25-65ms por utterance evaluada. Por eso NO
se llama por cada delta de R2T2 (ahí sigue habiendo keyword matching barato
para acciones locales/interrupción, que necesitan reacción instantánea) --
se llama UNA vez por turno, en `_escalate()`, donde 30-60ms es insignificante
contra el presupuesto de ~400-500ms del turno completo.
"""

from __future__ import annotations

from semantic_router import Route
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.routers import SemanticRouter

_ENCODER_MODEL_BY_LANG = {
    "English": "BAAI/bge-small-en-v1.5",
    "Spanish": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

_SCORE_THRESHOLD_BY_LANG = {
    "English": 0.5,  # default de FastEmbedEncoder, funcionó bien sin ajustar.
    "Spanish": 0.3,  # bajado de 0.5: el modelo multilingüe da similitudes
    # coseno más chicas incluso para frases claramente relacionadas --
    # con 0.5 el router devolvía None (sin ruta) para casos reales como
    # "explicame paso a paso cómo aprenden las redes neuronales".
}

_SLOW_PATH_UTTERANCES_EN = (
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

_FAST_PATH_UTTERANCES_EN = (
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

_SLOW_PATH_UTTERANCES_ES = (
    "podés explicarme eso en detalle",
    "dame un análisis profundo de esto",
    "explicame paso a paso cómo funciona",
    "compará los pros y los contras",
    "cuáles son las diferencias entre estas dos cosas",
    "contame más sobre la historia en profundidad",
    "dame un desglose completo",
    "hacé un análisis de las dos ideologías principales, capitalismo y socialismo",
    "podés profundizar en ese tema",
    "cuál es la diferencia entre capitalismo y socialismo",
    "explicame en detalle cómo funciona internet",
)

_FAST_PATH_UTTERANCES_ES = (
    "qué hora es",
    "encendé la luz",
    "hola cómo estás",
    "contame sobre la ciudad de méxico",
    "gracias",
    "continuá",
    "qué día es hoy",
    "apagá la luz",
    "cómo está el clima",
    "quién sos",
)

_UTTERANCES_BY_LANG = {
    "English": (_SLOW_PATH_UTTERANCES_EN, _FAST_PATH_UTTERANCES_EN),
    "Spanish": (_SLOW_PATH_UTTERANCES_ES, _FAST_PATH_UTTERANCES_ES),
}

_routers: dict[str, SemanticRouter] = {}


def _get_router(language: str) -> SemanticRouter:
    if language not in _routers:
        model_name = _ENCODER_MODEL_BY_LANG.get(language, _ENCODER_MODEL_BY_LANG["English"])
        threshold = _SCORE_THRESHOLD_BY_LANG.get(language, _SCORE_THRESHOLD_BY_LANG["English"])
        slow_utterances, fast_utterances = _UTTERANCES_BY_LANG.get(language, _UTTERANCES_BY_LANG["English"])
        encoder = FastEmbedEncoder(name=model_name)
        slow_path = Route(name="slow_path", utterances=list(slow_utterances), score_threshold=threshold)
        fast_path = Route(name="fast_path", utterances=list(fast_utterances), score_threshold=threshold)
        _routers[language] = SemanticRouter(encoder=encoder, routes=[slow_path, fast_path], auto_sync="local")
    return _routers[language]


def warm_up(language: str = "English") -> None:
    """Fuerza la carga del encoder + índice al arrancar el proceso, no en el
    primer turno real (evitaría un delay de ~1.5-2s en la primera escalada)."""
    _get_router(language)


def is_slow_path(text: str, language: str = "English") -> bool:
    result = _get_router(language)(text)
    return result.name == "slow_path"
