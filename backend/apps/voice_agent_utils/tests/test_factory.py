"""Tests for the provider factory + registry."""

import pytest

from voice_agent_utils import (
    STTConfig,
    TTSConfig,
    get_stt_provider,
    get_tts_provider,
)
from voice_agent_utils.exceptions import ProviderNotFoundError
from voice_agent_utils.providers.deepgram_stt import DeepgramSTTProvider
from voice_agent_utils.providers.deepgram_tts import DeepgramTTSProvider


class TestSTTFactory:
    def test_default_returns_deepgram(self):
        provider = get_stt_provider()
        assert isinstance(provider, DeepgramSTTProvider)
        assert provider.config.provider == "deepgram"

    def test_explicit_config_is_passed_through(self):
        cfg = STTConfig(provider="deepgram", model="nova-2", language="en")
        provider = get_stt_provider(cfg)
        assert provider.config is cfg
        assert provider.config.model == "nova-2"

    def test_unknown_provider_raises(self):
        with pytest.raises(ProviderNotFoundError):
            get_stt_provider(STTConfig(provider="does-not-exist"))


class TestTTSFactory:
    def test_default_returns_deepgram(self):
        provider = get_tts_provider()
        assert isinstance(provider, DeepgramTTSProvider)
        assert provider.config.provider == "deepgram"

    def test_explicit_config_is_passed_through(self):
        cfg = TTSConfig(provider="deepgram", model="aura-2-asteria-en")
        provider = get_tts_provider(cfg)
        assert provider.config is cfg
        assert provider.config.model == "aura-2-asteria-en"

    def test_unknown_provider_raises(self):
        with pytest.raises(ProviderNotFoundError):
            get_tts_provider(TTSConfig(provider="does-not-exist"))


class TestSettingsOverride:
    def test_settings_select_provider_when_no_config(self, settings):
        settings.VOICE_STT_PROVIDER = "does-not-exist"
        with pytest.raises(ProviderNotFoundError):
            get_stt_provider()

    def test_settings_tts_select_provider_when_no_config(self, settings):
        settings.VOICE_TTS_PROVIDER = "does-not-exist"
        with pytest.raises(ProviderNotFoundError):
            get_tts_provider()
