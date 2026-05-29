"""Unit tests for GeneralAssistant services (ImageModalHandler, VoiceModalHandler)."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from general_assistant.services import ImageModalHandler, VoiceModalHandler
from voice_agent_utils.base import BaseSTTProvider, BaseTTSProvider

pytestmark = pytest.mark.django_db


# ── ImageModalHandler.query_image_model ─────────────────────────────────────


class TestQueryImageModel:
    """Test the HuggingFace image-captioning via InferenceClient."""

    @patch("general_assistant.services._hf_client")
    async def test_success(self, mock_client):
        mock_result = MagicMock()
        mock_result.generated_text = "A dog running"
        mock_client.image_to_text.return_value = mock_result

        handler = ImageModalHandler()
        result = await handler.query_image_model(b"fake-bytes")

        assert result == "A dog running"
        mock_client.image_to_text.assert_called_once_with(b"fake-bytes")

    @patch("general_assistant.services._hf_client")
    async def test_api_error_returns_fallback(self, mock_client):
        mock_client.image_to_text.side_effect = ConnectionError("DNS failed")

        handler = ImageModalHandler()
        result = await handler.query_image_model(b"fake-bytes")

        assert "unavailable" in result.lower()

    @patch("general_assistant.services._hf_client")
    async def test_timeout_returns_fallback(self, mock_client):
        mock_client.image_to_text.side_effect = TimeoutError("timed out")

        handler = ImageModalHandler()
        result = await handler.query_image_model(b"fake-bytes")

        assert "unavailable" in result.lower()

    @patch("general_assistant.services._hf_client")
    async def test_unexpected_error_returns_fallback(self, mock_client):
        mock_client.image_to_text.side_effect = RuntimeError("something broke")

        handler = ImageModalHandler()
        result = await handler.query_image_model(b"fake-bytes")

        assert "unavailable" in result.lower()


# ── ImageModalHandler.process_image ─────────────────────────────────────────


class TestProcessImage:
    @patch("general_assistant.services.ImageModalHandler.query_image_model")
    async def test_returns_resized_image_and_description(self, mock_query):
        mock_query.return_value = "A beautiful sunset"

        handler = ImageModalHandler()
        fake_png = _make_png_bytes(200, 200)
        img_arr, description = await handler.process_image(fake_png)

        assert isinstance(img_arr, np.ndarray)
        assert img_arr.shape[1] == 720  # width resized to 720
        assert description == "A beautiful sunset"


# ── VoiceModalHandler ───────────────────────────────────────────────────────


class TestVoiceModalHandler:
    def test_init_wires_providers(self):
        handler = VoiceModalHandler()
        assert isinstance(handler._stt, BaseSTTProvider)
        assert isinstance(handler._tts, BaseTTSProvider)

    @patch("general_assistant.services.get_tts_provider")
    @patch("general_assistant.services.get_stt_provider")
    async def test_speech_to_text_delegates_to_provider(self, mock_stt, mock_tts):
        stt = MagicMock()

        async def _t(path):
            return "hello world"

        stt.transcribe_file.side_effect = _t
        mock_stt.return_value = stt
        mock_tts.return_value = MagicMock()

        handler = VoiceModalHandler()
        result = await handler.speech_to_text("/tmp/x.wav")

        assert result == "hello world"
        stt.transcribe_file.assert_called_once_with("/tmp/x.wav")

    @patch("general_assistant.services.get_tts_provider")
    @patch("general_assistant.services.get_stt_provider")
    async def test_text_to_speech_delegates_to_provider(self, mock_stt, mock_tts):
        tts = MagicMock()

        async def _s(text):
            return b"AUDIO"

        tts.synthesize.side_effect = _s
        mock_tts.return_value = tts
        mock_stt.return_value = MagicMock()

        handler = VoiceModalHandler()
        result = await handler.text_to_speech("Hi there")

        assert result == b"AUDIO"
        tts.synthesize.assert_called_once_with("Hi there")

    @patch("general_assistant.services.get_tts_provider")
    @patch("general_assistant.services.get_stt_provider")
    async def test_speech_to_text_swallows_provider_error(self, mock_stt, mock_tts):
        from voice_agent_utils import TranscriptionError

        stt = MagicMock()

        async def _t(path):
            raise TranscriptionError("boom")

        stt.transcribe_file.side_effect = _t
        mock_stt.return_value = stt
        mock_tts.return_value = MagicMock()

        handler = VoiceModalHandler()
        result = await handler.speech_to_text("/tmp/x.wav")

        assert "unavailable" in result.lower()

    @patch("general_assistant.services.get_tts_provider")
    @patch("general_assistant.services.get_stt_provider")
    async def test_text_to_speech_swallows_provider_error(self, mock_stt, mock_tts):
        from voice_agent_utils import SynthesisError

        tts = MagicMock()

        async def _s(text):
            raise SynthesisError("boom")

        tts.synthesize.side_effect = _s
        mock_tts.return_value = tts
        mock_stt.return_value = MagicMock()

        handler = VoiceModalHandler()
        result = await handler.text_to_speech("anything")

        assert result == b""


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_png_bytes(width, height):
    """Create minimal PNG bytes for testing (via cv2)."""
    import cv2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    success, buf = cv2.imencode(".png", img)
    assert success
    return buf.tobytes()
