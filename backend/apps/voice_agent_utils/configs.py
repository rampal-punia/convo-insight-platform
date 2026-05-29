"""Provider-agnostic STT / TTS configuration dataclasses.

These are intentionally generic — only fields that are meaningful across
multiple providers (Deepgram today; Bhashini, Cloudflare, Whisper, etc.
tomorrow) live here. Provider-specific knobs go into ``extra``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class STTConfig:
    """Configuration for a speech-to-text request."""

    provider: str = "deepgram"
    model: str = "nova-3"
    language: str = "en"
    sample_rate: int | None = None  # None = auto-detect from container
    encoding: str | None = None     # None = auto-detect from container
    punctuate: bool = True
    smart_format: bool = True
    diarize: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TTSConfig:
    """Configuration for a text-to-speech request."""

    provider: str = "deepgram"
    model: str = "aura-2-thalia-en"
    encoding: str = "mp3"
    container: str | None = None
    sample_rate: int | None = None
    speed: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
