"""Verificación multi-frase y multi-turno de NemotronASRService: confirma
que (a) no se pierde la última palabra con audio en ritmo real, (b) los
deltas llegan incrementales durante el turno, (c) flush_final() reabre el
turno correctamente para el siguiente (varios turnos en el mismo proceso).

Usa los WAVs ya sintetizados en tools/test_audio_fixtures/ -- no requiere
kokoro-fastapi corriendo.

Uso:
    python tools/smoke_nemotron_multi.py
"""

from __future__ import annotations

import asyncio
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "tools")

import numpy as np
import soundfile as sf

import audio_test_utils as base
from services.nemotron_stt import NemotronASRService, _get_model_and_processor

WAV_DIR = "tools/test_audio_fixtures"

CASES = [
    ("es_3.wav", "los medios de producción son de propiedad privada", "Spanish"),
    ("es_0.wav", "te estoy preguntando por la diferencia", "Spanish"),
    ("en_4.wav", "tell me the history of the city of Boston", "English"),
    ("en_6.wav", "turn on the light", "English"),
]


async def run_turn(svc: NemotronASRService, wav_name: str, expected: str) -> bool:
    samples, sr = sf.read(f"{WAV_DIR}/{wav_name}", dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples_16k = base.resample_linear(samples, sr, base.TARGET_SAMPLE_RATE)
    pcm16 = base.to_pcm16(samples_16k)

    deltas: list[str] = []
    chunk_ms = 160
    bytes_per_chunk = int(base.TARGET_SAMPLE_RATE * 2 * (chunk_ms / 1000.0))
    t0 = time.monotonic()
    delta_times: list[float] = []
    for i in range(0, len(pcm16), bytes_per_chunk):
        async for frame in svc.run_stt(pcm16[i : i + bytes_per_chunk]):
            deltas.append(frame.text)
            delta_times.append(time.monotonic() - t0)
        await asyncio.sleep(chunk_ms / 1000.0)  # ritmo real.

    tail = await svc.flush_final(timeout=3.0)
    deltas.append(tail)

    full = "".join(deltas)
    ok = base.normalize(full) == base.normalize(expected)
    audio_secs = len(samples_16k) / base.TARGET_SAMPLE_RATE
    incremental = len(delta_times) >= 2 and delta_times[-1] - delta_times[0] > 0.3

    print(f"[{'OK  ' if ok else 'FALLO'}] {wav_name}")
    print(f"    esperado:    {expected!r}")
    print(f"    transcrito:  {full!r}")
    print(
        f"    deltas: {len(delta_times)} durante el turno "
        f"(primero t={delta_times[0]:.2f}s, último t={delta_times[-1]:.2f}s "
        f"de {audio_secs:.2f}s de audio) -> incremental={incremental}"
    )
    print(f"    cola de flush_final: {tail!r}\n")
    return ok


async def main() -> None:
    print("Cargando modelo...")
    _get_model_and_processor()

    passed = 0
    for wav_name, expected, language in CASES:
        svc = NemotronASRService(language=language, chunk_size_ms=160)
        svc._loop = asyncio.get_running_loop()
        svc._sample_rate = base.TARGET_SAMPLE_RATE
        svc._user_id = "smoke"
        svc._open_turn()

        passed += await run_turn(svc, wav_name, expected)

        # Segundo turno con la MISMA instancia: verifica que flush_final()
        # dejó el servicio en estado usable (reapertura de turno correcta).
        print(f"    -- segundo turno seguido con la misma instancia ({wav_name}) --")
        passed_again = await run_turn(svc, wav_name, expected)
        if not passed_again:
            print("    !! la reapertura de turno rompió la transcripción\n")

        svc._close_turn()

    print(f"Resultado: {passed}/{len(CASES)} primeros turnos exactos.")


if __name__ == "__main__":
    asyncio.run(main())
