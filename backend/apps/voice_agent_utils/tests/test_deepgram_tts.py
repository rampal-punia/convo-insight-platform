"""Tests for the Deepgram TTS provider (network mocked)."""

import pytest
from unittest.mock import MagicMock, patch

from voice_agent_utils import SynthesisError, TTSConfig
from voice_agent_utils.providers.deepgram_tts import DeepgramTTSProvider


class TestDeepgramTTSProvider:
    @patch("voice_agent_utils.providers.deepgram_tts.DeepgramClient")
    async def test_synthesize_joins_chunks(self, mock_client_cls, settings):
        settings.DEEPGRAM_API_KEY = "test-key"

        client = MagicMock()
        client.speak.v1.audio.generate.return_value = iter([b"abc", b"def", b""])
        mock_client_cls.return_value = client

        provider = DeepgramTTSProvider(TTSConfig())
        audio = await provider.synthesize("Hello")

        assert audio == b"abcdef"
        kwargs = client.speak.v1.audio.generate.call_args.kwargs
        assert kwargs["text"] == "Hello"
        assert kwargs["model"] == "aura-2-thalia-en"
        assert kwargs["encoding"] == "mp3"

    @patch("voice_agent_utils.providers.deepgram_tts.DeepgramClient")
    async def test_synthesize_accepts_bytes_response(self, mock_client_cls, settings):
        settings.DEEPGRAM_API_KEY = "test-key"

        client = MagicMock()
        client.speak.v1.audio.generate.return_value = b"BLOB"
        mock_client_cls.return_value = client

        provider = DeepgramTTSProvider(TTSConfig())
        audio = await provider.synthesize("Hi")

        assert audio == b"BLOB"

    async def test_missing_api_key_raises(self, settings):
        settings.DEEPGRAM_API_KEY = ""
        provider = DeepgramTTSProvider(TTSConfig())
        with pytest.raises(SynthesisError, match="DEEPGRAM_API_KEY"):
            await provider.synthesize("hi")

    async def test_empty_text_raises(self, settings):
        settings.DEEPGRAM_API_KEY = "test-key"
        provider = DeepgramTTSProvider(TTSConfig())
        with pytest.raises(SynthesisError, match="Empty"):
            await provider.synthesize("   ")

    @patch("voice_agent_utils.providers.deepgram_tts.DeepgramClient")
    async def test_sdk_exception_wraps(self, mock_client_cls, settings):
        settings.DEEPGRAM_API_KEY = "test-key"

        client = MagicMock()
        client.speak.v1.audio.generate.side_effect = RuntimeError("boom")
        mock_client_cls.return_value = client

        provider = DeepgramTTSProvider(TTSConfig())
        with pytest.raises(SynthesisError, match="boom"):
            await provider.synthesize("x")

    @patch("voice_agent_utils.providers.deepgram_tts.DeepgramClient")
    async def test_optional_kwargs_passed_through(self, mock_client_cls, settings):
        settings.DEEPGRAM_API_KEY = "test-key"

        client = MagicMock()
        client.speak.v1.audio.generate.return_value = b"x"
        mock_client_cls.return_value = client

        cfg = TTSConfig(
            encoding="linear16",
            container="wav",
            sample_rate=24000,
            speed=1.2,
            extra={"tag": "test"},
        )
        provider = DeepgramTTSProvider(cfg)
        await provider.synthesize("Hi")

        kwargs = client.speak.v1.audio.generate.call_args.kwargs
        assert kwargs["encoding"] == "linear16"
        assert kwargs["container"] == "wav"
        assert kwargs["sample_rate"] == 24000
        assert kwargs["speed"] == 1.2
        assert kwargs["tag"] == "test"
