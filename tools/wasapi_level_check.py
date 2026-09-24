import pyaudio
import numpy as np
import time
import sys

p = pyaudio.PyAudio()
idx = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    host_api = p.get_host_api_info_by_index(info["hostApi"])["name"]
    if "Realtek USB Audio" in info["name"] and "WASAPI" in host_api and info["maxInputChannels"] > 0:
        idx = i
        break

if idx is None:
    print("NO SE ENCONTRO EL DISPOSITIVO WASAPI")
    sys.exit(1)

info = p.get_device_info_by_index(idx)
print(f"Abriendo indice {idx}: {info['name']} @ {info['defaultSampleRate']}Hz")

rate = int(info["defaultSampleRate"])
stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                 input_device_index=idx, frames_per_buffer=rate // 10)

print("Capturando 6 segundos, HABLA AHORA fuerte y claro cerca del mic...")
t0 = time.time()
peaks = []
while time.time() - t0 < 6:
    data = stream.read(rate // 10, exception_on_overflow=False)
    arr = np.frombuffer(data, dtype=np.int16)
    rms = np.sqrt(np.mean(arr.astype(np.float64) ** 2))
    peak = np.max(np.abs(arr)) if len(arr) else 0
    peaks.append(peak)
    print(f"  t={time.time()-t0:4.1f}s  RMS={rms:8.1f}  peak={peak:6d}  (max int16=32767)")

stream.stop_stream()
stream.close()
p.terminate()

print(f"\nPico maximo detectado en toda la captura: {max(peaks)}")
if max(peaks) < 200:
    print("SEÑAL PRACTICAMENTE SILENCIADA -- EchoNull/Equalizer APO esta anulando el mic.")
elif max(peaks) < 2000:
    print("SEÑAL MUY DEBIL -- posible atenuacion agresiva.")
else:
    print("SEÑAL TIENE NIVEL NORMAL -- el problema NO es el nivel de captura crudo.")
