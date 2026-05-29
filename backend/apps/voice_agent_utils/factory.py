"""Provider registry + factory functions.

Adding a new provider (e.g. Bhashini, Cloudflare, Whisper):

    1. Implement ``BaseSTTProvider`` / ``BaseTTSProvider`` in
       ``providers/<name>_stt.py`` / ``providers/<name>_tts.py``.
    2. Register the class in ``_STT_REGISTRY`` / ``_TTS_REGISTRY`` below.
    3. Flip ``settings.VOICE_STT_PROVIDER`` / ``settings.VOICE_TTS_PROVIDER``.

That is the entire migration.
"""

from __future__ import annotations

from typing import Callable

from django.conf import settings

from .base import BaseSTTProvider, BaseTTSProvider
from .configs import STTConfig, TTSConfig
from .exceptions import ProviderNotFoundError
from .providers.deepgram_stt import DeepgramSTTProvider
from .providers.deepgram_tts import DeepgramTTSProvider

_STT_REGISTRY: dict[str, Callable[[STTConfig], BaseSTTProvider]] = {
    "deepgram": DeepgramSTTProvider,
}

_TTS_REGISTRY: dict[str, Callable[[TTSConfig], BaseTTSProvider]] = {
    "deepgram": DeepgramTTSProvider,
}


def _resolve_name(config_name: str | None, settings_attr: str, default: str) -> str:
    if config_name:
        return config_name
    return getattr(settings, settings_attr, default)


def get_stt_provider(config: STTConfig | None = None) -> BaseSTTProvider:
    """Return a configured STT provider.

    If ``config`` is omitted, the provider name is read from
    ``settings.VOICE_STT_PROVIDER`` (default ``"deepgram"``).
    """
    name = _resolve_name(
        getattr(config, "provider", None) if config else None,
        "VOICE_STT_PROVIDER",
        "deepgram",
    )
    try:
        provider_cls = _STT_REGISTRY[name]
    except KeyError as exc:
        raise ProviderNotFoundError(
            f"Unknown STT provider '{name}'. "
            f"Registered: {sorted(_STT_REGISTRY)}"
        ) from exc
    cfg = config or STTConfig(provider=name)
    return provider_cls(cfg)


def get_tts_provider(config: TTSConfig | None = None) -> BaseTTSProvider:
    """Return a configured TTS provider.

    If ``config`` is omitted, the provider name is read from
    ``settings.VOICE_TTS_PROVIDER`` (default ``"deepgram"``).
    """
    name = _resolve_name(
        getattr(config, "provider", None) if config else None,
        "VOICE_TTS_PROVIDER",
        "deepgram",
    )
    try:
        provider_cls = _TTS_REGISTRY[name]
    except KeyError as exc:
        raise ProviderNotFoundError(
            f"Unknown TTS provider '{name}'. "
            f"Registered: {sorted(_TTS_REGISTRY)}"
        ) from exc
    cfg = config or TTSConfig(provider=name)
    return provider_cls(cfg)
