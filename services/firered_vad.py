"""Adaptador Pipecat de FireRedVAD (streaming) como reemplazo de Silero VAD.

FireRedVAD (https://huggingface.co/FireRedTeam/FireRedVAD) no es un VADAnalyzer
nativo de Pipecat: es un paquete standalone (`fireredvad`) con su propio modelo
DFSMN streaming. Este wrapper implementa la interfaz abstracta de Pipecat
(`pipecat.audio.vad.vad_analyzer.VADAnalyzer`) delegando la inferencia por
frame a `fireredvad.FireRedStreamVad.detect_chunk`.

Requisitos (no están en PyPI, instalar desde el repo oficial):
    git clone https://github.com/FireRedTeam/FireRedVAD.git
    pip install -r FireRedVAD/requirements.txt
    huggingface-cli download FireRedTeam/FireRedVAD --local-dir ./pretrained_models/FireRedVAD
    export PYTHONPATH=$PWD/FireRedVAD:$PYTHONPATH
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

try:
    from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "Para usar FireRedVAD instala el paquete `fireredvad` desde "
        "https://github.com/FireRedTeam/FireRedVAD (no está en PyPI)."
    )
    raise ImportError(f"Missing module: {e}") from e

# fireredvad.core.constants: 16 kHz, ventana (frame_length) 25 ms, salto 10 ms.
# La ventana mínima que el extractor kaldi-fbank necesita para emitir al
# menos 1 frame es FRAME_LENGTH_SAMPLE (25 ms); pedirle menos (p.ej. el hop
# de 10 ms) revienta con "zero channel inputs" porque no completa ni una
# ventana de análisis.
_FRAME_LENGTH_SAMPLE = 400  # 25 ms a 16 kHz.


class FireRedVADAnalyzer(VADAnalyzer):
    """VADAnalyzer de Pipecat respaldado por el modelo streaming de FireRedVAD."""

    def __init__(
        self,
        *,
        model_dir: str = "./pretrained_models/FireRedVAD/Stream-VAD",
        use_gpu: bool = False,
        speech_threshold: float = 0.4,
        sample_rate: int | None = 16000,
        params: VADParams | None = None,
    ):
        super().__init__(sample_rate=sample_rate, params=params)

        logger.debug(f"[FireRedVAD] Cargando modelo streaming desde {model_dir}...")
        config = FireRedStreamVadConfig(
            use_gpu=use_gpu,
            speech_threshold=speech_threshold,
        )
        self._stream_vad = FireRedStreamVad.from_pretrained(model_dir, config)
        self._last_confidence = 0.0
        logger.debug("[FireRedVAD] Modelo cargado.")

    def set_sample_rate(self, sample_rate: int):
        if sample_rate != 16000:
            raise ValueError(
                f"FireRedVAD requiere 16000 Hz (sample rate recibido: {sample_rate})"
            )
        super().set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        # 25 ms por llamada: mínimo que el extractor de features necesita
        # para producir al menos un frame válido.
        return _FRAME_LENGTH_SAMPLE

    def voice_confidence(self, buffer: bytes) -> float:
        try:
            # `extract()` espera amplitud int16 real (-32768..32767), NO
            # normalizada a [-1,1]: el ejemplo oficial (ws_server.py) le pasa
            # el array int16 directo. Normalizar acá (como hacíamos antes)
            # producía features ~32768x más chicas de lo esperado y el
            # modelo nunca reportaba confianza > 0.02 pese a haber voz real.
            audio_int16 = np.frombuffer(buffer, dtype=np.int16)

            frame_results = self._stream_vad.detect_chunk(audio_int16)
            if frame_results:
                # El extractor puede emitir 0 o varios frames de 25 ms por
                # cada hop de 10 ms recibido; nos quedamos con el más reciente.
                self._last_confidence = frame_results[-1].smoothed_prob

            return self._last_confidence
        except Exception as e:
            logger.error(f"Error analizando audio con FireRedVAD: {e}")
            return 0.0

    async def cleanup(self):
        self._stream_vad.reset()
        await super().cleanup()
