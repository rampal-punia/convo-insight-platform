from django.core.files.base import ContentFile
from django.conf import settings
from PIL import Image
import numpy as np
import cv2
import io
import requests
import asyncio
import librosa
import speech_recognition as sr
from gtts import gTTS
import tempfile
import soundfile as sf

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": f"Bearer {settings.HUGGINGFACEHUB_API_TOKEN}"}


class VoiceModalHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    async def process_audio(self, message):
        audio_file = message.audio_content.audio_file.path
        text = await self.speech_to_text(audio_file)
        duration = await self.get_audio_duration(audio_file)
        await self.update_audio_message(message.audio_content, text, duration)
        return text

    async def speech_to_text(self, audio_file):
        def recognize():
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
            try:
                return self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "Speech recognition could not understand the audio"
            except sr.RequestError:
                return "Could not request results from the speech recognition service"

        return await asyncio.to_thread(recognize)

    async def text_to_speech(self, text):
        def generate_audio():
            tts = gTTS(text=text, lang='en')
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            return temp_file.name

        audio_file = await asyncio.to_thread(generate_audio)
        with open(audio_file, 'rb') as f:
            audio_content = f.read()
        return audio_content

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
        response = await asyncio.to_thread(
            requests.post,
            API_URL,
            headers=headers,
            data=image_bytes
        )
        return response.json()[0]['generated_text']

    @staticmethod
    def update_image_message(image_message, image_array, image_description):
        print(
            type(image_message),
            type(image_array),
            type(image_description),
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


if __name__ == '__main__':
    wav_file = '/home/ram/convo-insight-platform/media/voice_messages/audio_11.wav'
    vh = VoiceModalHandler()

    # Run the async function using asyncio.run()
    # text = asyncio.run(vh.speech_to_text(wav_file))
    # print(text)

    text = "Life is awesome and programming in python is great."
    speech = asyncio.run(vh.text_to_speech(text))
    print(speech)
