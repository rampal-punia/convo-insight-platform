"""Pluggable voice (STT / TTS) provider layer.

The rest of the application should only depend on:

    from voice_agent_utils import get_stt_provider, get_tts_provider
    from voice_agent_utils import STTConfig, TTSConfig
    from voice_agent_utils import TranscriptionError, SynthesisError

Swapping Deepgram for Bhashini / Cloudflare / Whisper later means
adding a new provider class under ``providers/`` and registering it
in ``factory._REGISTRY`` — no callers change.
"""

from .configs import STTConfig, TTSConfig
from .exceptions import (
    SynthesisError,
    TranscriptionError,
    VoiceProviderError,
)
from .factory import get_stt_provider, get_tts_provider

__all__ = [
    "STTConfig",
    "TTSConfig",
    "SynthesisError",
    "TranscriptionError",
    "VoiceProviderError",
    "get_stt_provider",
    "get_tts_provider",
]
