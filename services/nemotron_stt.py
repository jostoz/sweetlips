"""Cliente Pipecat para Nemotron 3.5 ASR (NVIDIA, cache-aware
FastConformer-RNNT streaming) -- reemplazo de R2T2 tras confirmar en vivo
(ver README, sección "Migración ASR: R2T2 -> Nemotron") que R2T2 trunca la
última palabra de una fracción significativa de turnos, en ambos idiomas,
con código 100% upstream sin modificar -- no es un bug nuestro, es la
arquitectura append-only del modelo.

A diferencia de R2T2 (servidor WebSocket separado en WSL2, porque vLLM no
corre nativo en Windows), Nemotron corre en-proceso en Windows vía
🤗 Transformers (AutoModelForRNNT) -- sin WSL2, sin servidor aparte.

Puente de arquitectura: `model.generate(streamer=..., input_features=<gen>)`
de Transformers es una API "pull" (el modelo jala chunks de un generador
Python síncrono hasta que se agota) que corre en un thread aparte, y
`TextIteratorStreamer` empuja texto de vuelta también desde ese thread. Pero
`STTService.run_stt()` de pipecat es "push" y asíncrono (pipecat empuja
bytes de audio a medida que llegan del mic). Este servicio puentea ambos
mundos con un `queue.Queue` (thread-safe) para audio entrante y un
`asyncio.Queue` (alimentado vía `call_soon_threadsafe`) para texto saliente.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService

_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
_NUM_LOOKAHEAD_TOKENS = 6
"""~560ms de latencia (medido). Debe coincidir entre
processor.set_num_lookahead_tokens() (fija el tamaño de chunk/contexto
derecho) y el num_lookahead_tokens= pasado a generate() -- el modelo
valida que coincidan y tira ValueError si no."""

# Modelo + processor son pesados (carga de pesos ~3s, ~2.9GB VRAM) --
# compartidos entre turnos/instancias del proceso, cargados una sola vez.
_model = None
_processor = None
_load_lock = threading.Lock()


def _get_model_and_processor():
    global _model, _processor
    with _load_lock:
        if _model is None:
            logger.info(f"[Nemotron] Cargando modelo {_MODEL_ID} (GPU)...")
            _processor = AutoProcessor.from_pretrained(_MODEL_ID)
            _model = AutoModelForRNNT.from_pretrained(_MODEL_ID, device_map="cuda")
            # 6 tokens de lookahead = ~560ms de latencia (medido). Fijo al
            # cargar porque el processor es compartido (singleton de
            # módulo) -- no varía entre turnos ni idiomas.
            _processor.set_num_lookahead_tokens(_NUM_LOOKAHEAD_TOKENS)
            logger.info("[Nemotron] Modelo cargado.")
    return _model, _processor


_LANG_MAP = {
    "English": "en-US",
    "Spanish": "es-ES",
}


class NemotronASRService(STTService):
    """STT streaming para Nemotron 3.5 ASR (cache-aware FastConformer-RNNT)."""

    def __init__(
        self,
        language: str = "English",
        chunk_size_ms: int = 160,
        **kwargs,
    ):
        super().__init__(
            settings=STTSettings(model="nemotron-3.5-asr-streaming-0.6b", language=language),
            **kwargs,
        )
        self.language = language
        self._target_lang = _LANG_MAP.get(language, "en-US")
        self.chunk_size_ms = chunk_size_ms

        self._audio_queue: queue.Queue[bytes | None] | None = None
        self._text_queue: asyncio.Queue[str] | None = None
        self._gen_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_buffer = bytearray()

    async def setup(self, setup) -> None:
        await super().setup(setup)
        self._loop = asyncio.get_running_loop()
        # Carga el modelo en un executor -- bloquea ~3-5s, no queremos
        # trabar el event loop del pipeline en el arranque.
        await self._loop.run_in_executor(None, _get_model_and_processor)
        self._open_turn()

    async def cleanup(self) -> None:
        self._close_turn()
        await super().cleanup()

    def _bytes_per_chunk(self) -> int:
        return int(self.sample_rate * 2 * (self.chunk_size_ms / 1000.0))

    def _open_turn(self) -> None:
        """Arranca un nuevo generate() en background para el próximo turno."""
        model, processor = _get_model_and_processor()
        self._audio_queue = queue.Queue()
        self._text_queue = asyncio.Queue()

        def build_first_inputs_and_rest():
            """Devuelve (first_inputs_dict_completo, generador_de_chunks_restantes).
            A diferencia de una versión anterior que solo pasaba
            `input_features` a generate(), acá exponemos el dict COMPLETO
            del primer chunk (incluye prompt_ids/attention_mask que el
            processor arma a partir de `language=`) -- generate() sin eso
            cae a detección automática de idioma y (bug real medido) corta
            la última palabra del turno silenciosamente."""
            sample_rate = processor.feature_extractor.sampling_rate
            sample_buf = np.zeros(0, dtype=np.float32)
            exhausted = False

            def pull_until(n: int) -> None:
                nonlocal sample_buf, exhausted
                while len(sample_buf) < n and not exhausted:
                    raw = self._audio_queue.get()
                    if raw is None:
                        exhausted = True
                        return
                    chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    sample_buf = np.concatenate([sample_buf, chunk])

            pull_until(processor.num_samples_first_audio_chunk)
            if len(sample_buf) < processor.num_samples_first_audio_chunk:
                return None, None  # turno cerrado antes de juntar ni el primer chunk.

            first_inputs = processor(
                sample_buf[: processor.num_samples_first_audio_chunk],
                sampling_rate=sample_rate,
                is_streaming=True,
                is_first_audio_chunk=True,
                language=self._target_lang,
                return_tensors="pt",
            )
            first_inputs = first_inputs.to(model.device, dtype=model.dtype)
            first_inputs["input_features"] = first_inputs["input_features"][
                :, : processor.num_mel_frames_first_audio_chunk, :
            ]

            mel_frame_idx = processor.num_mel_frames_first_audio_chunk
            hop_length = processor.feature_extractor.hop_length
            n_fft = processor.feature_extractor.n_fft
            start_idx = mel_frame_idx * hop_length - n_fft // 2

            def rest_generator():
                nonlocal mel_frame_idx, start_idx
                while True:
                    end_idx = start_idx + processor.num_samples_per_audio_chunk
                    pull_until(end_idx)
                    if len(sample_buf) < end_idx:
                        if exhausted and start_idx < len(sample_buf):
                            # Cola corta final (menos de un chunk completo,
                            # p.ej. la última palabra del turno + silencio
                            # de cierre): el código fuente de transformers
                            # (generation_nemotron_asr_streaming.py) exige
                            # chunks de tamaño EXACTO y documenta "pad the
                            # final chunk if needed" -- sin esto se
                            # descarta en silencio y se pierde audio real
                            # (bug medido: "...propiedad privada" ->
                            # "...propiedad ", perdiendo la última
                            # palabra).
                            tail = sample_buf[start_idx:]
                            padded = np.zeros(processor.num_samples_per_audio_chunk, dtype=np.float32)
                            padded[: len(tail)] = tail
                            inputs = processor(
                                padded,
                                sampling_rate=sample_rate,
                                is_streaming=True,
                                is_first_audio_chunk=False,
                                language=self._target_lang,
                                return_tensors="pt",
                            )
                            yield inputs.input_features.to(model.device, dtype=model.dtype)
                        return
                    inputs = processor(
                        sample_buf[start_idx:end_idx],
                        sampling_rate=sample_rate,
                        is_streaming=True,
                        is_first_audio_chunk=False,
                        language=self._target_lang,
                        return_tensors="pt",
                    )
                    yield inputs.input_features.to(model.device, dtype=model.dtype)
                    mel_frame_idx += processor.num_mel_frames_per_audio_chunk
                    start_idx = mel_frame_idx * hop_length - n_fft // 2

            return first_inputs, rest_generator()

        def run_generate():
            streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
            first_inputs, rest_gen = build_first_inputs_and_rest()
            if first_inputs is None:
                return

            def resumed_generator():
                yield first_inputs["input_features"]
                yield from rest_gen

            def drain_streamer():
                # Debe correr en un thread APARTE del que llama a
                # generate(): TextIteratorStreamer bloquea hasta que haya
                # texto nuevo o generate() termine -- si lo iterás en el
                # mismo thread que generate() (después del call), el texto
                # llega todo junto al final, no incremental (streaming
                # falso). Iterarlo en paralelo es lo que da deltas reales.
                for text_chunk in streamer:
                    if text_chunk and self._loop is not None:
                        self._loop.call_soon_threadsafe(
                            self._text_queue.put_nowait, text_chunk
                        )

            drain_thread = threading.Thread(target=drain_streamer, daemon=True)
            drain_thread.start()

            generate_kwargs = dict(first_inputs)
            generate_kwargs["input_features"] = resumed_generator()
            generate_kwargs["streamer"] = streamer
            generate_kwargs["num_lookahead_tokens"] = _NUM_LOOKAHEAD_TOKENS
            try:
                model.generate(**generate_kwargs)
            except Exception:
                import traceback
                logger.error(f"[Nemotron] generate() interrumpido:\n{traceback.format_exc()}")
            drain_thread.join(timeout=5.0)
        self._gen_thread = threading.Thread(target=run_generate, daemon=True)
        self._gen_thread.start()
        logger.info(f"[Nemotron] Turno abierto (idioma={self._target_lang}).")


    _CLOSING_SILENCE_SECS = 1.2
    """Silencio que se empuja al cerrar el turno. El streaming cache-aware
    consume chunks de tamaño EXACTO (num_samples_per_audio_chunk = 585ms
    con lookahead 6): si el audio del usuario termina a mitad de un chunk,
    ese resto NO forma un chunk válido y el modelo nunca llega a decodificar
    las últimas palabras (bug medido: "...propiedad privada" ->
    "...propiedad ", faltando 8451 muestras para cerrar el último chunk).
    Empujar silencio real -- no padding artificial de un slice parcial, que
    probamos y no alcanzó -- completa ese chunk y deja que el decoder RNNT
    emita lo que tenía pendiente. 1.2s cubre dos chunks completos con
    margen."""

    def _close_turn(self, timeout: float = 5.0) -> None:
        if self._audio_queue is not None:
            # Audio pendiente en _audio_buffer (menos de un chunk
            # completo, ver run_stt) NUNCA se manda solo -- si no lo
            # empujamos acá antes del sentinel, se pierde en silencio y
            # el generador nunca ve los últimos ~160ms de audio.
            if self._audio_buffer:
                self._audio_queue.put(bytes(self._audio_buffer))
            silence_samples = int(self.sample_rate * self._CLOSING_SILENCE_SECS)
            self._audio_queue.put(b"\x00\x00" * silence_samples)
            self._audio_queue.put(None)
        if self._gen_thread is not None:
            self._gen_thread.join(timeout=timeout)
            self._gen_thread = None
        self._audio_buffer.clear()

    async def flush_final(self, timeout: float = 1.2) -> str:
        """Cierra el turno actual (fuerza a Nemotron a finalizar la
        transcripción con el audio pendiente), drena el texto final, y
        abre el próximo turno -- mismo contrato que ConfuciusR2T2Service
        (ver services/r2t2_stt.py) para no tener que tocar jev_system1.py."""
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: self._close_turn(timeout=timeout)
        )
        # El thread de drenaje entrega texto con call_soon_threadsafe, que
        # solo AGENDA el put_nowait en el event loop -- no lo ejecuta. Sin
        # ceder el control acá, drenaríamos _text_queue con los callbacks
        # del final del turno todavía pendientes y perderíamos la última
        # palabra (bug medido: "...propiedad privada" -> "...propiedad ").
        # Solo se notaba con audio en tiempo real: con audio precargado los
        # tokens llegaban mucho antes, entre los await del loop de entrada.
        await asyncio.sleep(0.05)

        final_chunks: list[str] = []
        while not self._text_queue.empty():
            final_chunks.append(self._text_queue.get_nowait())

        self._open_turn()
        return "".join(final_chunks)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        self._audio_buffer.extend(audio)

        if self._audio_queue is None:
            return

        bytes_needed = self._bytes_per_chunk()
        while len(self._audio_buffer) >= bytes_needed:
            chunk = bytes(self._audio_buffer[:bytes_needed])
            del self._audio_buffer[:bytes_needed]
            self._audio_queue.put(chunk)

        while not self._text_queue.empty():
            text = self._text_queue.get_nowait()
            yield TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    async def _process_assistant_turn(self, text: str) -> None:
        self._close_turn()
        self._open_turn()
