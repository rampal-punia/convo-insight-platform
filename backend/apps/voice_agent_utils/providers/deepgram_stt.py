"""Deepgram speech-to-text provider (prerecorded ``listen.v1`` endpoint).

Wraps the synchronous Deepgram SDK in ``asyncio.to_thread`` so it can be
called from Django Channels consumers without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from deepgram import DeepgramClient
from django.conf import settings

from ..base import BaseSTTProvider
from ..exceptions import TranscriptionError

logger = logging.getLogger("voice_agent_utils.deepgram_stt")


class DeepgramSTTProvider(BaseSTTProvider):
    """Deepgram prerecorded transcription."""

    def _build_client(self) -> Any:
        api_key = getattr(settings, "DEEPGRAM_API_KEY", "") or ""
        if not api_key:
            raise TranscriptionError(
                "DEEPGRAM_API_KEY is not configured in Django settings."
            )
        return DeepgramClient(api_key=api_key)

    def _build_kwargs(self) -> dict[str, Any]:
        cfg = self.config
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "language": cfg.language,
            "punctuate": cfg.punctuate,
            "smart_format": cfg.smart_format,
            "diarize": cfg.diarize,
        }
        if cfg.sample_rate is not None:
            kwargs["sample_rate"] = cfg.sample_rate
        if cfg.encoding is not None:
            kwargs["encoding"] = cfg.encoding
        kwargs.update(cfg.extra or {})
        return kwargs

    @staticmethod
    def _extract_transcript(response: Any) -> str:
        try:
            return (
                response.results.channels[0].alternatives[0].transcript or ""
            ).strip()
        except (AttributeError, IndexError, TypeError) as exc:
            raise TranscriptionError(
                f"Unexpected Deepgram response shape: {exc}"
            ) from exc

    async def transcribe_bytes(self, audio: bytes) -> str:
        if not audio:
            raise TranscriptionError("Empty audio payload.")

        def _call() -> str:
            try:
                client = self._build_client()
                response = client.listen.v1.media.transcribe_file(
                    request=audio,
                    **self._build_kwargs(),
                )
            except TranscriptionError:
                raise
            except Exception as exc:  # noqa: BLE001 — provider-agnostic boundary
                logger.exception("Deepgram STT failed")
                raise TranscriptionError(str(exc)) from exc
            return self._extract_transcript(response)

        return await asyncio.to_thread(_call)
