import asyncio
import librosa
import speech_recognition as sr
from gtts import gTTS
import tempfile
import soundfile as sf


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


if __name__ == '__main__':
    wav_file = '/home/ram/convo-insight-platform/media/voice_messages/audio_11.wav'
    vh = VoiceModalHandler()

    # Run the async function using asyncio.run()
    # text = asyncio.run(vh.speech_to_text(wav_file))
    # print(text)

    text = "Life is awesome and programming in python is great."
    speech = asyncio.run(vh.text_to_speech(text))
    print(speech)
