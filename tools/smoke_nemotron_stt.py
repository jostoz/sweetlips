"""Smoke test directo de NemotronASRService, sin pipecat/pipeline completo
-- confirma que el puente de threading (audio_chunk_generator +
TextIteratorStreamer + drain concurrente) realmente entrega deltas
incrementales, no solo texto al final, y que flush_final()/reconexión de
turno funcionan como en R2T2.

Uso:
    python tools/smoke_nemotron_stt.py
"""

from __future__ import annotations

import asyncio
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "tools")

from services.nemotron_stt import NemotronASRService
import audio_test_utils as base


class _FakeSetup:
    pass


async def main() -> None:
    svc = NemotronASRService(language="Spanish", chunk_size_ms=160)
    # STTService.setup() espera un objeto FrameProcessorSetup real en
    # pipecat -- para este smoke test llamamos directo a lo que
    # necesitamos sin pasar por todo pipecat.
    from services.nemotron_stt import _get_model_and_processor

    print("Cargando modelo...")
    t0 = time.monotonic()
    _get_model_and_processor()
    print(f"Modelo cargado en {time.monotonic() - t0:.1f}s")

    svc._loop = asyncio.get_running_loop()
    svc._sample_rate = base.TARGET_SAMPLE_RATE
    svc._user_id = "smoke-test"
    svc._open_turn()

    phrase = "los medios de producción son de propiedad privada"
    print(f'\nProbando: "{phrase}"')
    samples, src_rate = await base.synthesize(phrase)
    samples_16k = base.resample_linear(samples, src_rate, base.TARGET_SAMPLE_RATE)
    pcm16 = base.to_pcm16(samples_16k)

    deltas: list[str] = []
    chunk_ms = 160
    bytes_per_chunk = int(base.TARGET_SAMPLE_RATE * 2 * (chunk_ms / 1000.0))
    t_start = time.monotonic()
    for i in range(0, len(pcm16), bytes_per_chunk):
        chunk = pcm16[i : i + bytes_per_chunk]
        async for frame in svc.run_stt(chunk):
            deltas.append(frame.text)
            print(f"  [t={time.monotonic() - t_start:.2f}s] delta: {frame.text!r}")
        await asyncio.sleep(chunk_ms / 1000.0)

    print("\nLlamando flush_final()...")
    tail = await svc.flush_final(timeout=10.0)
    print(f"  tail: {tail!r}")
    deltas.append(tail)

    full_text = "".join(deltas)
    print(f"\nTexto completo reconstruido: {full_text!r}")
    print(f"Esperado: {phrase!r}")
    print(f"Match (normalizado): {base.normalize(full_text) == base.normalize(phrase)}")

    svc._close_turn()


if __name__ == "__main__":
    asyncio.run(main())
