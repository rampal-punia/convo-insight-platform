import logging

from django.core.files.base import ContentFile
from django.conf import settings
from PIL import Image
import numpy as np
import cv2
import io
import asyncio
import librosa
import soundfile as sf
from huggingface_hub import InferenceClient

from voice_agent_utils import (
    SynthesisError,
    TranscriptionError,
    get_stt_provider,
    get_tts_provider,
)

logger = logging.getLogger('general_assistant')

_hf_client = InferenceClient(
    model='Salesforce/blip-image-captioning-large',
    token=settings.HUGGINGFACEHUB_API_TOKEN,
)


class VoiceModalHandler:
    """Audio pipeline backed by the pluggable ``voice_agent_utils`` providers."""

    def __init__(self):
        self._stt = get_stt_provider()
        self._tts = get_tts_provider()

    async def process_audio(self, message):
        audio_file = message.audio_content.audio_file.path
        text = await self.speech_to_text(audio_file)
        duration = await self.get_audio_duration(audio_file)
        await self.update_audio_message(message.audio_content, text, duration)
        return text

    async def speech_to_text(self, audio_file):
        try:
            return await self._stt.transcribe_file(audio_file)
        except TranscriptionError as exc:
            logger.warning("STT failed: %s", exc)
            return "Speech recognition is currently unavailable."

    async def text_to_speech(self, text):
        try:
            return await self._tts.synthesize(text)
        except SynthesisError as exc:
            logger.warning("TTS failed: %s", exc)
            return b""

    async def get_audio_duration(self, audio_file):
        def get_duration():
            y, sr = librosa.load(audio_file)
            return librosa.get_duration(y=y, sr=sr)

        return await asyncio.to_thread(get_duration)

    @staticmethod
    async def update_audio_message(audio_message, transcript, duration):
        audio_message.transcript = transcript
        audio_message.duration = duration
        await asyncio.to_thread(audio_message.save)

    @staticmethod
    async def convert_audio_format(input_file, output_file, target_sr=22050):
        def convert():
            # Load the audio file
            y, sr = librosa.load(input_file, sr=None)

            # Resample if necessary
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

            # Write the audio file
            sf.write(output_file, y, target_sr)

        await asyncio.to_thread(convert)


class ImageModalHandler:
    def __init__(self):
        pass

    async def process_image(self, image_content):
        # Convert image content to numpy array
        img_arr = np.frombuffer(image_content, np.int8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        # Resize image
        img_resized = cv2.resize(img, (720, int(720*9/16)))

        # Convert back to bytes
        is_success, im_buf_arr = cv2.imencode(".png", img_resized)
        byte_im = im_buf_arr.tobytes()

        # Get image description
        image_description = await self.query_image_model(byte_im)

        return img_resized, image_description

    async def query_image_model(self, image_bytes):
        try:
            result = await asyncio.to_thread(_hf_client.image_to_text, image_bytes)
            return result.generated_text if hasattr(result, 'generated_text') else str(result)
        except Exception as exc:
            logger.warning("Image captioning API unavailable: %s", exc)
            return "Image uploaded successfully, but the image description service is currently unavailable."

    @staticmethod
    def update_image_message(image_message, image_array, image_description):
        logger.debug(
            "Updating image message: msg=%s, arr=%s, desc=%s",
            type(image_message).__name__,
            type(image_array).__name__,
            type(image_description).__name__,
        )
        # Convert numpy array to PIL Image
        img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

        # Save to BytesIO object
        img_io = io.BytesIO()
        img.save(img_io, format='PNG')

        # Create a Django ContentFile
        image_content = ContentFile(
            img_io.getvalue(), name=f"image_{image_message.id}.png")

        # Update ImageMessage
        image_message.image = image_content
        image_message.width, image_message.height = img.size
        image_message.description = image_description
        image_message.save()



