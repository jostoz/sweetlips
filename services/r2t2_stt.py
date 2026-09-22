"""Cliente WebSocket de Pipecat para Confucius4-R2T2 (ASR streaming append-only).

vLLM no soporta Windows nativo (confirmado: `pip install vllm` cae a compilar
el sdist, sin wheel binario). Por eso R2T2 corre como servidor dentro de WSL2
(Linux real, GPU passthrough via nvidia-smi verificado) usando el
`ws_server.py` que trae el propio repo
(https://github.com/netease-youdao/Confucius4-R2T2), y este servicio de
Pipecat en Windows le habla por WebSocket (`/asr_stream_api_v1`).

Protocolo confirmado leyendo `ws_server.py`/`ws_client.py` del repo:
    1. Conectar a ``ws://<host>:<port>/asr_stream_api_v1``.
    2. Enviar un header JSON primero:
       {"requestId": str, "secret_key": ..., "language": "zhen"/"Chinese"/...,
        "use_vad": bool, "mode": "slow"|"fast"}
    3. El servidor responde {"status": "connected", ...}.
    4. Enviar audio PCM16 mono 16 kHz como mensajes binarios (bytes crudos).
    5. El servidor responde por cada chunk procesado:
       {"status": "success", "msg": {"text": <DELTA>, "reset": bool, ...}}
       ``text`` ya es el delta (texto nuevo estable), no acumulado — no hace
       falta diffing en el cliente.
    6. Al cerrar el turno, enviar el string literal
       ``YOUDAO_ONETIME_ASR_STREAM_EOS``; el servidor manda el delta final y
       cierra la conexión (hay que reconectar para el siguiente turno).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from loguru import logger
from websockets.asyncio.client import connect as ws_connect

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService

_EOS = "YOUDAO_ONETIME_ASR_STREAM_EOS"


class ConfuciusR2T2Service(STTService):
    """STT streaming para Confucius4-R2T2 vía el servidor WebSocket del repo oficial."""

    def __init__(
        self,
        ws_uri: str = "ws://localhost:8272/asr_stream_api_v1",
        secret_key: str = "test0102",
        language: str = "zhen",
        chunk_size_ms: int = 160,
        use_vad: bool = False,
        **kwargs,
    ):
        super().__init__(
            settings=STTSettings(model="Confucius4-R2T2", language=language),
            **kwargs,
        )
        self.ws_uri = ws_uri
        self.secret_key = secret_key
        self.language = language
        self.chunk_size_ms = chunk_size_ms
        self.use_vad = use_vad

        self._audio_buffer = bytearray()
        self._ws = None
        self._receiver_task: asyncio.Task | None = None
        self._pending: asyncio.Queue[str] = asyncio.Queue()
        self._closing = False

    async def setup(self, setup) -> None:
        await super().setup(setup)
        await self._open_turn()

    async def cleanup(self) -> None:
        await self._close_turn()
        await super().cleanup()

    async def _open_turn(self) -> None:
        """Abre una nueva conexión WebSocket + turno de streaming en el servidor."""
        logger.info(f"[R2T2] Conectando a {self.ws_uri}...")
        self._ws = await ws_connect(self.ws_uri)
        header = {
            "requestId": str(uuid.uuid4()),
            "secret_key": self.secret_key,
            "language": self.language,
            "use_vad": self.use_vad,
            "mode": "slow",
        }
        await self._ws.send(json.dumps(header))

        ack_raw = await self._ws.recv()
        ack = json.loads(ack_raw)
        if ack.get("status") != "connected":
            raise RuntimeError(f"[R2T2] Conexión rechazada por el servidor: {ack}")
        logger.info(f"[R2T2] Turno conectado (requestId={header['requestId']}).")

        self._closing = False
        self._receiver_task = asyncio.create_task(self._receiver_loop())

    async def _close_turn(self) -> None:
        self._closing = True
        if self._ws is not None:
            try:
                await self._ws.send(_EOS)
            except Exception:
                pass
        if self._receiver_task is not None:
            try:
                await asyncio.wait_for(self._receiver_task, timeout=10.0)
            except (asyncio.TimeoutError, Exception):
                self._receiver_task.cancel()
            self._receiver_task = None
        self._ws = None

    async def flush_final(self, timeout: float = 0.6) -> str:
        """Fuerza a R2T2 a emitir cualquier delta que haya quedado
        procesando del audio ya enviado, antes de que Jev decida el texto
        final del turno.

        Sin esto, Jev escala con lo que R2T2 alcanzó a mandar hasta el
        instante exacto en que el VAD detecta silencio -- pero el
        encoder/decoder de R2T2 tiene su propia latencia de inferencia, así
        que la última palabra dicha suele quedar "en vuelo" y se pierde
        (medido en vivo: "tu lugar favor[ito]", "tú no me escuch[aste]").

        Protocolo (ver docstring del módulo): mandar el string literal EOS
        fuerza al servidor a mandar el delta final y cerrar la conexión.
        Cierra la conexión actual, drena lo que haya llegado, y reabre.

        Nota: se probó NO reabrir acá (delegarle la reconexión al hook
        `_process_assistant_turn` de STTService, esperando que dispare
        después de la respuesta del bot) para darle más tiempo de
        "calentamiento" a la conexión nueva -- causó una regresión MUCHO
        peor: ese hook nunca dispara en esta arquitectura (usamos
        System2PromptBridge/ResponseCollector propios, no el
        LLMContextAggregatorPair estándar de pipecat del que probablemente
        depende), así que R2T2 quedaba desconectado para siempre después
        del primer turno -- el bot dejaba de contestar por completo. Se
        revierte a reabrir siempre acá, que es menos elegante pero
        confiable.
        """
        if self._ws is None:
            return ""
        self._closing = True
        try:
            await self._ws.send(_EOS)
        except Exception:
            pass
        if self._receiver_task is not None:
            try:
                await asyncio.wait_for(self._receiver_task, timeout=timeout)
            except (asyncio.TimeoutError, Exception):
                self._receiver_task.cancel()
            self._receiver_task = None
        self._ws = None
        self._closing = False

        final_chunks: list[str] = []
        while not self._pending.empty():
            final_chunks.append(self._pending.get_nowait())

        await self._open_turn()
        return "".join(final_chunks)

    async def _receiver_loop(self) -> None:
        """Lee mensajes del servidor y encola los deltas de texto no vacíos."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if not isinstance(raw, str):
                    continue
                msg = json.loads(raw)
                if msg.get("status") not in ("success",):
                    continue
                text = msg.get("msg", {}).get("text", "")
                if text:
                    await self._pending.put(text)
        except Exception as e:
            if not self._closing:
                logger.error(f"[R2T2] Receptor WebSocket interrumpido: {e}")

    def _bytes_per_chunk(self) -> int:
        # 16 kHz, 16 bit (2 bytes), mono.
        return int(self.sample_rate * 2 * (self.chunk_size_ms / 1000.0))

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        self._audio_buffer.extend(audio)

        if self._ws is None:
            # Reconexión en curso (ver flush_final): en vez de tirar error
            # y perder este chunk, lo dejamos en el buffer -- se manda
            # apenas la conexión nueva esté lista, en la próxima llamada.
            return

        bytes_needed = self._bytes_per_chunk()

        while len(self._audio_buffer) >= bytes_needed:
            chunk = bytes(self._audio_buffer[:bytes_needed])
            del self._audio_buffer[:bytes_needed]
            await self._ws.send(chunk)

        # Drenar cualquier delta que ya haya llegado de forma asíncrona
        # (el servidor procesa y responde en paralelo al envío de audio).
        while not self._pending.empty():
            text = self._pending.get_nowait()
            yield TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    async def _process_assistant_turn(self, text: str) -> None:
        # Fin de turno: cerrar la conexión actual (el servidor manda el delta
        # final tras el EOS) y abrir una nueva para el próximo turno del usuario.
        await self._close_turn()
        await self._open_turn()
