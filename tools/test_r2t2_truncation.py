"""Script de reproducción para el bug de truncamiento de R2T2.

Sintetiza frases de prueba conocidas por disparar el bug (terminan justo
después de una palabra larga, o en un patrón que ya cortó texto en vivo
hoy: "la ciudad de", "gracias, continúa", "socialismo y capitalismo"...),
las manda directo al servidor R2T2 (WebSocket, sin pasar por micrófono
real ni por pipecat), y compara la transcripción reconstruida contra el
texto esperado -- útil para verificar en frío el fix de `flush_final()`
(timeout 0.6s -> 1.2s, ver services/r2t2_stt.py) sin depender de hablarle
al pipeline en vivo cada vez.

Requiere: kokoro-fastapi corriendo en :8880, r2t2-ws-server en :8272
(ver README para arrancar ambos).

Uso:
    python tools/test_r2t2_truncation.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid

import httpx
import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect as ws_connect

R2T2_WS_URI = "ws://127.0.0.1:8272/asr_stream_api_v1"
KOKORO_URL = "http://127.0.0.1:8880/v1/audio/speech"
KOKORO_VOICE = "ef_dora"  # español -- cambiar a "af_heart" si se prueba en inglés.
TARGET_SAMPLE_RATE = 16000
CHUNK_MS = 160  # mismo chunk_size_ms que usa ConfuciusR2T2Service por default.
EOS = "YOUDAO_ONETIME_ASR_STREAM_EOS"

# Frases que reprodujeron el bug de truncamiento hoy en vivo (terminan justo
# después de una palabra sustancial, sin silencio de sobra al final -- el
# caso exacto donde el lookahead de 320ms de R2T2 se come el final si
# flush_final() no espera lo suficiente).
TEST_PHRASES = [
    "te estoy preguntando por la diferencia",
    "gracias, continúa por favor",
    "cuál es la diferencia entre socialismo y capitalismo",
    "los medios de producción son de propiedad privada",
    "contame la historia de la ciudad de méxico",
    "qué hora es",  # frase corta de control, no debería fallar nunca.
    "prendé la luz",  # otra frase corta de control.
]


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


async def transcribe(pcm16_audio: bytes, language: str = "Spanish") -> str:
    """Manda el audio a R2T2 en chunks, como lo haría el pipeline real, y
    devuelve el texto reconstruido completo (deltas + flush final)."""
    header = {
        "requestId": str(uuid.uuid4()),
        "secret_key": "test0102",
        "language": language,
        "use_vad": False,
        "mode": "slow",
    }
    chunks: list[str] = []
    async with ws_connect(R2T2_WS_URI) as ws:
        await ws.send(json.dumps(header))
        connected = json.loads(await ws.recv())
        if connected.get("status") != "connected":
            raise RuntimeError(f"R2T2 no confirmó conexión: {connected}")

        bytes_per_chunk = int(TARGET_SAMPLE_RATE * 2 * (CHUNK_MS / 1000.0))

        async def receiver():
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    text = msg.get("msg", {}).get("text", "")
                    if text:
                        chunks.append(text)
            except Exception:
                pass

        recv_task = asyncio.create_task(receiver())

        for i in range(0, len(pcm16_audio), bytes_per_chunk):
            await ws.send(pcm16_audio[i : i + bytes_per_chunk])
            await asyncio.sleep(CHUNK_MS / 1000.0)  # ritmo real (1x), no acelerado.

        # Simula el margen natural del pipeline real ANTES de llamar a
        # flush_final(): VAD stop_secs (0.4s) + inferencia de smart-turn
        # (~15-90ms) + grace period tras COMPLETE (0.35s, ver
        # _SMART_TURN_COMPLETE_GRACE_SECS en jev_system1.py) -- sin este
        # margen, el test es MÁS agresivo que el caso real (manda EOS
        # apenas termina el audio, sin darle tiempo a R2T2 de procesar lo
        # que ya tiene en cola antes de pedirle el flush).
        await asyncio.sleep(0.8)

        # Igual que flush_final(): EOS fuerza el delta final antes de cerrar.
        await ws.send(EOS)
        try:
            await asyncio.wait_for(recv_task, timeout=1.2)
        except asyncio.TimeoutError:
            recv_task.cancel()

    return "".join(chunks)


def normalize(text: str) -> str:
    return " ".join(text.lower().replace(",", "").replace(".", "").split())


async def run_case(phrase: str) -> bool:
    samples, src_rate = await synthesize(phrase)
    samples_16k = resample_linear(samples, src_rate, TARGET_SAMPLE_RATE)
    pcm16 = to_pcm16(samples_16k)
    transcribed = await transcribe(pcm16)

    expected_norm = normalize(phrase)
    got_norm = normalize(transcribed)
    ok = got_norm.endswith(expected_norm.split()[-1]) if expected_norm else False

    status = "OK  " if ok else "FAIL"
    print(f"[{status}] esperado:     {phrase!r}")
    print(f"        transcrito:   {transcribed!r}")
    if not ok:
        print("        -> la ULTIMA palabra esperada no aparece completa al final")
    print()
    return ok


async def main():
    print(f"Probando {len(TEST_PHRASES)} frases contra R2T2 ({R2T2_WS_URI})...\n")
    results = []
    for phrase in TEST_PHRASES:
        try:
            results.append(await run_case(phrase))
        except Exception as e:
            print(f"[ERROR] {phrase!r}: {e}\n")
            results.append(False)

    passed = sum(results)
    total = len(results)
    print(f"Resultado: {passed}/{total} frases sin truncar al final.")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
