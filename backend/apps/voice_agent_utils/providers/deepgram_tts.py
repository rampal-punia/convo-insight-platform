"""Deepgram text-to-speech provider (``speak.v1.audio.generate``).

``generate`` returns an iterator of audio bytes; we concatenate to a
single ``bytes`` payload for callers that just want a finished blob
(e.g. saving as a file or sending over WebSocket).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from django.conf import settings

from ..base import BaseTTSProvider
from ..exceptions import SynthesisError

logger = logging.getLogger("voice_agent_utils.deepgram_tts")


class DeepgramTTSProvider(BaseTTSProvider):
    """Deepgram Aura TTS."""

    def _build_client(self) -> Any:
        from deepgram import DeepgramClient

        api_key = getattr(settings, "DEEPGRAM_API_KEY", "") or ""
        if not api_key:
            raise SynthesisError(
                "DEEPGRAM_API_KEY is not configured in Django settings."
            )
        return DeepgramClient(api_key=api_key)

    def _build_kwargs(self, text: str) -> dict[str, Any]:
        cfg = self.config
        kwargs: dict[str, Any] = {"text": text, "model": cfg.model}
        if cfg.encoding is not None:
            kwargs["encoding"] = cfg.encoding
        if cfg.container is not None:
            kwargs["container"] = cfg.container
        if cfg.sample_rate is not None:
            kwargs["sample_rate"] = cfg.sample_rate
        if cfg.speed is not None:
            kwargs["speed"] = cfg.speed
        kwargs.update(cfg.extra or {})
        return kwargs

    async def synthesize(self, text: str) -> bytes:
        if not text or not text.strip():
            raise SynthesisError("Empty text payload.")

        def _call() -> bytes:
            try:
                client = self._build_client()
                stream = client.speak.v1.audio.generate(
                    **self._build_kwargs(text)
                )
                if isinstance(stream, (bytes, bytearray)):
                    return bytes(stream)
                # SDK returns Iterator[bytes]; join all chunks.
                return b"".join(chunk for chunk in stream if chunk)
            except SynthesisError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Deepgram TTS failed")
                raise SynthesisError(str(exc)) from exc

        return await asyncio.to_thread(_call)
