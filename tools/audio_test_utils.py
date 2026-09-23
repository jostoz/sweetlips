"""Helpers de audio para los smoke tests del pipeline: sintetiza frases con
Kokoro-FastAPI, resamplea a 16 kHz y convierte a PCM16 -- lo que necesitan
los tests de ASR para alimentar audio realista sin depender del micrófono.

Extraído de tools/test_r2t2_truncation.py (el runner específico de R2T2 se
borró junto con ese motor; los resultados de esa investigación quedaron
documentados en el README).

Requiere kokoro-fastapi corriendo en :8880 SOLO para synthesize(); el resto
de los helpers son puro numpy.
"""

from __future__ import annotations

import io

import httpx
import numpy as np
import soundfile as sf

KOKORO_URL = "http://127.0.0.1:8880/v1/audio/speech"
KOKORO_VOICE = "ef_dora"  # español; "af_heart" para inglés.
TARGET_SAMPLE_RATE = 16000

async def synthesize(text: str) -> tuple[np.ndarray, int]:
    """Llama a Kokoro-FastAPI y devuelve (samples float32, sample_rate).
    Agrega 500ms de silencio de cola: audio sintético corta abrupto al
    final del texto, pero un micrófono real sigue capturando ambiente
    (aire, respiración) un rato después de que la persona termina de
    hablar -- sin ese margen, el test es artificialmente peor que el
    caso real."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            KOKORO_URL,
            json={"model": "kokoro", "input": text, "voice": KOKORO_VOICE, "response_format": "wav"},
        )
        resp.raise_for_status()
        import io

        data, sample_rate = sf.read(io.BytesIO(resp.content), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # downmix a mono si viene estéreo.
        silence = np.zeros(int(sample_rate * 1.0), dtype=np.float32)
        data = np.concatenate([data, silence])
        return data, sample_rate


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample simple por interpolación lineal -- alcanza para este test
    (no hace falta calidad audiófila, solo que R2T2 pueda transcribir)."""
    if src_rate == dst_rate:
        return samples
    duration = len(samples) / src_rate
    dst_n = int(duration * dst_rate)
    src_x = np.linspace(0, duration, num=len(samples), endpoint=False)
    dst_x = np.linspace(0, duration, num=dst_n, endpoint=False)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def normalize(text: str) -> str:
    return " ".join(text.lower().replace(",", "").replace(".", "").split())


