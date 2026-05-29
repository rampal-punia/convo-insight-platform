"""Abstract base classes for voice providers.

All concrete providers (Deepgram, Bhashini, Cloudflare, …) implement
these interfaces. The application layer only ever sees these ABCs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .configs import STTConfig, TTSConfig


class BaseSTTProvider(ABC):
    """Speech-to-text provider interface."""

    def __init__(self, config: STTConfig) -> None:
        self.config = config

    @abstractmethod
    async def transcribe_bytes(self, audio: bytes) -> str:
        """Transcribe raw audio bytes. Returns the transcript text."""

    async def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file on disk. Default impl reads + delegates."""
        with open(path, "rb") as f:
            data = f.read()
        return await self.transcribe_bytes(data)


class BaseTTSProvider(ABC):
    """Text-to-speech provider interface."""

    def __init__(self, config: TTSConfig) -> None:
        self.config = config

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize ``text`` into audio bytes (encoded per config)."""
