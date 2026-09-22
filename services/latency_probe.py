"""Medición de latencia del pipeline: fin de turno -> texto escalado -> LLM
arranca -> LLM termina -> primer audio del bot.

Dos salidas en paralelo:
  - print() a stdout (diagnóstico rápido en logs, como antes).
  - Histogramas Prometheus (observabilidad real: p50/p95/p99 por etapa,
    dashboard en Grafana). Servidor HTTP de métricas se levanta una sola
    vez con `start_metrics_server()`, llamado desde main.py al arrancar
    el pipeline.

Las etapas (labels) coinciden con los `mark()` que ya existían en
jev_system1.py y system2_llm.py -- no requirió tocar los call sites,
sólo agregar la observación Prometheus dentro de `mark()`.
"""

from __future__ import annotations

import time

from prometheus_client import Histogram, start_http_server

_t0: float | None = None

# Buckets en segundos: pensados para latencia de voz conversacional
# (decenas de ms a pocos segundos). Coincide con lo medido en esta sesión:
# LLM first-token ~5-400ms, LLM completo ~300-500ms, audio ~0.5-2.7s.
_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)

STAGE_LATENCY = Histogram(
    "voice_pipeline_stage_latency_seconds",
    "Tiempo desde que un turno escala a System 2 hasta cada etapa del pipeline.",
    labelnames=("stage",),
    buckets=_BUCKETS,
)

_metrics_server_started = False


def start_metrics_server(port: int = 9091) -> None:
    """Levanta el endpoint /metrics (Prometheus text format) una sola vez.

    Llamar al arrancar el pipeline, antes de correr el `PipelineRunner`.
    Scrapeado por Prometheus en `observability/prometheus.yml`.
    """
    global _metrics_server_started
    if _metrics_server_started:
        return
    start_http_server(port)
    _metrics_server_started = True
    print(f"[LAT] Métricas Prometheus en http://127.0.0.1:{port}/metrics", flush=True)


def mark_turn_start() -> None:
    """Llamar cuando Jev decide escalar (texto final del turno listo)."""
    global _t0
    _t0 = time.monotonic()
    print("[LAT] t=0ms   turno escalado", flush=True)


def mark(label: str) -> None:
    if _t0 is None:
        return
    elapsed_s = time.monotonic() - _t0
    print(f"[LAT] t={elapsed_s * 1000:.0f}ms {label}", flush=True)
    STAGE_LATENCY.labels(stage=label).observe(elapsed_s)
