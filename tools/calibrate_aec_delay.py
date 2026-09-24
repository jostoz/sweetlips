"""Calibración de delay del AEC: mismo principio que un micrófono de
calibración acústica (ej. Denon DM-A409/Audyssey) -- en vez de asumir o
adivinar cuánto tarda el sonido en ir del parlante al micrófono, se
reproduce una señal CONOCIDA (chirp) y se mide el delay real por
correlación cruzada contra lo que efectivamente capta el mic físico.

Esto reemplaza `stream_delay_ms=0` (dejar que el estimador interno de
AEC3 lo adivine) por un valor medido -- útil si el estimador interno no
converge bien solo (ver README, investigación AEC de hoy: ERLE negativo
en ~15-20% de los picos incluso con el buffer far-end ya alineado).

Requiere: mic real (mismo input_device_index que usa main.py) y
parlantes/auriculares conectados, en silencio ambiente.

Uso:
    python tools/calibrate_aec_delay.py
"""

from __future__ import annotations

import numpy as np
import pyaudiowpatch as pyaudio

_SAMPLE_RATE = 16000
_MIC_DEVICE_INDEX = 1  # mismo índice que main.py (Realtek USB Audio, MME).
_CHIRP_DURATION_SECS = 0.5
_SILENCE_PAD_SECS = 0.3
_RECORD_SECS = 2.0  # suficiente para capturar el chirp + margen.


def _make_chirp() -> np.ndarray:
    """Chirp lineal 500Hz->4000Hz: banda ancha, buena autocorrelación (pico
    angosto), y audible/distinguible de ruido de fondo típico."""
    t = np.linspace(0, _CHIRP_DURATION_SECS, int(_SAMPLE_RATE * _CHIRP_DURATION_SECS), endpoint=False)
    f0, f1 = 500, 4000
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * _CHIRP_DURATION_SECS)))
    # Envelope corto para evitar clicks al principio/final (ringing que
    # ensucia la correlación).
    envelope = np.ones_like(chirp)
    ramp = int(0.02 * _SAMPLE_RATE)
    envelope[:ramp] = np.linspace(0, 1, ramp)
    envelope[-ramp:] = np.linspace(1, 0, ramp)
    chirp = chirp * envelope * 0.7
    silence = np.zeros(int(_SAMPLE_RATE * _SILENCE_PAD_SECS))
    return np.concatenate([silence, chirp, silence]).astype(np.float32)


def _measure_delay(played: np.ndarray, recorded: np.ndarray) -> float:
    """Correlación cruzada normalizada: el offset del pico es el delay
    real (en muestras) entre que se mandó la señal y que el mic la captó."""
    correlation = np.correlate(recorded, played, mode="full")
    peak_idx = int(np.argmax(np.abs(correlation)))
    lag_samples = peak_idx - (len(played) - 1)
    peak_value = float(np.abs(correlation[peak_idx]))
    noise_floor = float(np.median(np.abs(correlation)))
    snr = peak_value / (noise_floor + 1e-6)
    return lag_samples / _SAMPLE_RATE * 1000, snr


def main() -> None:
    chirp = _make_chirp()
    print(f"Chirp generado: {len(chirp) / _SAMPLE_RATE:.2f}s")

    pa = pyaudio.PyAudio()
    try:
        recorded_chunks: list[bytes] = []
        record_frames = int(_SAMPLE_RATE * _RECORD_SECS)

        def _rec_callback(in_data, frame_count, time_info, status):
            recorded_chunks.append(in_data)
            return (None, pyaudio.paContinue)

        rec_stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=_SAMPLE_RATE,
            input=True,
            input_device_index=_MIC_DEVICE_INDEX,
            frames_per_buffer=1024,
            stream_callback=_rec_callback,
        )

        play_stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=_SAMPLE_RATE,
            output=True,
        )

        print("Grabando + reproduciendo chirp de calibración (silencio ambiente, por favor)...")
        rec_stream.start_stream()
        import time

        time.sleep(0.1)  # asegura que la grabación ya esté activa antes de reproducir.
        play_stream.write(chirp.tobytes())
        play_stream.stop_stream()
        play_stream.close()

        while sum(len(c) for c in recorded_chunks) < record_frames * 4 and rec_stream.is_active():
            time.sleep(0.05)

        rec_stream.stop_stream()
        rec_stream.close()

        recorded = np.frombuffer(b"".join(recorded_chunks), dtype=np.float32)
        print(f"Grabado: {len(recorded) / _SAMPLE_RATE:.2f}s")

        delay_ms, snr = _measure_delay(chirp, recorded)
        print(f"\nDelay medido: {delay_ms:.1f}ms (SNR de correlación: {snr:.1f}x)")
        if snr < 3.0:
            print(
                "AVISO: SNR bajo -- el pico de correlación no es claro. Puede que el "
                "chirp no se haya escuchado bien (volumen bajo, ruido ambiente) o que "
                "el mic/parlante no coincidan con los configurados en main.py."
            )
        if delay_ms < 0:
            print(
                "AVISO: delay negativo no tiene sentido físico (el mic no puede "
                "captar el sonido ANTES de que se reproduzca) -- probablemente "
                "ruido de fondo dominó la correlación, no el chirp real."
            )
        else:
            print(
                f"\nUsar en main.py: WebRTCAECFilter(far_end_buffer, "
                f"stream_delay_ms={round(delay_ms)})"
            )
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
