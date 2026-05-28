"""Unit tests for GeneralAssistant services (ImageModalHandler, VoiceModalHandler)."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from general_assistant.services import ImageModalHandler, VoiceModalHandler

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
    def test_init(self):
        handler = VoiceModalHandler()
        assert handler.recognizer is not None


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_png_bytes(width, height):
    """Create minimal PNG bytes for testing (via cv2)."""
    import cv2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    success, buf = cv2.imencode(".png", img)
    assert success
    return buf.tobytes()
