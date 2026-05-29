"""Tests for the Deepgram STT provider (network mocked)."""

import pytest
from unittest.mock import MagicMock, patch

from voice_agent_utils import STTConfig, TranscriptionError
from voice_agent_utils.providers.deepgram_stt import DeepgramSTTProvider


def _fake_response(transcript: str) -> MagicMock:
    response = MagicMock()
    response.results.channels[0].alternatives[0].transcript = transcript
    return response


class TestDeepgramSTTProvider:
    @patch("voice_agent_utils.providers.deepgram_stt.DeepgramClient")
    async def test_transcribe_bytes_returns_transcript(self, mock_client_cls, settings):
        settings.DEEPGRAM_API_KEY = "test-key"

        client = MagicMock()
        client.listen.v1.media.transcribe_file.return_value = _fake_response(
            "hello world"
        )
        mock_client_cls.return_value = client

        provider = DeepgramSTTProvider(STTConfig())
        result = await provider.transcribe_bytes(b"\x00\x01\x02")

        assert result == "hello world"
        mock_client_cls.assert_called_once_with(api_key="test-key")
        call_kwargs = client.listen.v1.media.transcribe_file.call_args.kwargs
        assert call_kwargs["request"] == b"\x00\x01\x02"
        assert call_kwargs["model"] == "nova-3"
        assert call_kwargs["language"] == "en"
        assert call_kwargs["punctuate"] is True
        assert call_kwargs["smart_format"] is True

    @patch("voice_agent_utils.providers.deepgram_stt.DeepgramClient")
    async def test_transcribe_file_reads_then_delegates(
        self, mock_client_cls, settings, tmp_path
    ):
        settings.DEEPGRAM_API_KEY = "test-key"

        client = MagicMock()
        client.listen.v1.media.transcribe_file.return_value = _fake_response("ok")
        mock_client_cls.return_value = client

        audio_path = tmp_path / "a.wav"
        audio_path.write_bytes(b"WAVDATA")

        provider = DeepgramSTTProvider(STTConfig())
        result = await provider.transcribe_file(str(audio_path))

        assert result == "ok"
        assert (
            client.listen.v1.media.transcribe_file.call_args.kwargs["request"]
            == b"WAVDATA"
        )

    async def test_missing_api_key_raises(self, settings):
        settings.DEEPGRAM_API_KEY = ""
        provider = DeepgramSTTProvider(STTConfig())
        with pytest.raises(TranscriptionError, match="DEEPGRAM_API_KEY"):
            await provider.transcribe_bytes(b"x")

    async def test_empty_audio_raises(self, settings):
        settings.DEEPGRAM_API_KEY = "test-key"
        provider = DeepgramSTTProvider(STTConfig())
        with pytest.raises(TranscriptionError, match="Empty"):
            await provider.transcribe_bytes(b"")

    @patch("voice_agent_utils.providers.deepgram_stt.DeepgramClient")
    async def test_sdk_exception_wraps_to_transcription_error(
        self, mock_client_cls, settings
    ):
        settings.DEEPGRAM_API_KEY = "test-key"

        client = MagicMock()
        client.listen.v1.media.transcribe_file.side_effect = RuntimeError("boom")
        mock_client_cls.return_value = client

        provider = DeepgramSTTProvider(STTConfig())
        with pytest.raises(TranscriptionError, match="boom"):
            await provider.transcribe_bytes(b"x")

    @patch("voice_agent_utils.providers.deepgram_stt.DeepgramClient")
    async def test_extra_kwargs_passed_through(self, mock_client_cls, settings):
        settings.DEEPGRAM_API_KEY = "test-key"

        client = MagicMock()
        client.listen.v1.media.transcribe_file.return_value = _fake_response("x")
        mock_client_cls.return_value = client

        cfg = STTConfig(
            sample_rate=16000,
            encoding="linear16",
            extra={"keyterm": "Convo"},
        )
        provider = DeepgramSTTProvider(cfg)
        await provider.transcribe_bytes(b"x")

        kwargs = client.listen.v1.media.transcribe_file.call_args.kwargs
        assert kwargs["sample_rate"] == 16000
        assert kwargs["encoding"] == "linear16"
        assert kwargs["keyterm"] == "Convo"
