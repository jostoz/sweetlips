"""Instrumentación temporal mínima para medir dónde se va la latencia real
del pipeline: fin de turno -> texto escalado -> LLM arranca -> LLM termina
-> primer audio del bot.

No es una librería de métricas seria (no exporta a Prometheus/etc.) — es
un helper de una sola sesión para diagnosticar antes de optimizar a ciegas.
"""

from __future__ import annotations

import time

_t0: float | None = None


def mark_turn_start() -> None:
    """Llamar cuando Jev decide escalar (texto final del turno listo)."""
    global _t0
    _t0 = time.monotonic()
    print("[LAT] t=0ms   turno escalado", flush=True)


def mark(label: str) -> None:
    if _t0 is None:
        return
    elapsed_ms = (time.monotonic() - _t0) * 1000
    print(f"[LAT] t={elapsed_ms:.0f}ms {label}", flush=True)
