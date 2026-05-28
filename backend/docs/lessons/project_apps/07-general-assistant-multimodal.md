# Lesson 7: General Assistant — Multimodal AI

> How the general assistant handles text, image, and voice inputs through WebSocket.

---

## What You'll Learn

- Multimodal AI (text + image + voice in one consumer)
- Image processing with OpenCV + PIL
- Image captioning with HuggingFace InferenceClient
- Speech-to-text with Google Speech Recognition
- Text-to-speech with gTTS
- Audio processing with librosa
- Graceful error handling for external API failures

---

## 1. The Multimodal Consumer

The `GeneralAssistantConsumer` is a single WebSocket consumer that handles three types of input:

```
WebSocket Connection
       │
       ├──► text_message   → LangChain LLM chain → streaming response
       ├──► image_message  → Image processing + HuggingFace captioning
       └──► audio_message  → Speech-to-text → LangChain LLM chain
```

### Consumer Structure (`apps/general_assistant/consumers.py`):

```python
class GeneralAssistantConsumer(AsyncJsonWebsocketConsumer):
    async def receive_json(self, content, **kwargs):
        message_type = content.get('type')

        if message_type == 'text_message':
            await self.handle_text_message(content)

        elif message_type == 'image_message':
            await self.handle_image_message(content)

        elif message_type == 'audio_message':
            await self.handle_audio_message(content)
```

### Message routing:

| `type` field | Handler | Input | Output |
|-------------|---------|-------|--------|
| `text_message` | `handle_text_message` | Text string | Streaming AI response |
| `image_message` | `handle_image_message` | Base64 image | Image description + AI response |
| `audio_message` | `handle_audio_message` | Base64 audio | Transcript + AI response |

---

## 2. Text Message Flow

The simplest path — same as the ConvoChat lesson:

```
"text_message" → Save to DB → Build chain with history → Stream LLM response → Save AI response
```

```python
async def handle_text_message(self, content):
    text = content.get('content', '').strip()
    conversation = await self._get_or_create_conversation()

    # Save user message
    user_msg = await self._save_user_message(conversation, text)

    # Stream AI response
    await TextChatHandler.process_text_response(
        conversation=conversation,
        user_message=user_msg,
        input_data=text,
        send_method=self.send
    )
```

---

## 3. Image Processing Pipeline

### Flow:

```
Base64 string → Decode bytes → OpenCV resize → HuggingFace caption → Description text
      │                                                  │
      │                                          "A dog running in a park"
      │
      ▼
   Resize to 720px width
   Save processed image to Django media storage
```

### ImageModalHandler (`apps/general_assistant/services.py`):

```python
class ImageModalHandler:
    async def process_image(self, image_content):
        # Step 1: Decode raw bytes to numpy array
        img_arr = np.frombuffer(image_content, np.int8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        # Step 2: Resize to standard dimensions (720px width)
        img_resized = cv2.resize(img, (720, int(720 * 9 / 16)))

        # Step 3: Encode back to bytes for API call
        is_success, im_buf_arr = cv2.imencode(".png", img_resized)
        byte_im = im_buf_arr.tobytes()

        # Step 4: Get image description from HuggingFace
        image_description = await self.query_image_model(byte_im)

        return img_resized, image_description
```

### Image Captioning with HuggingFace:

```python
from huggingface_hub import InferenceClient

_hf_client = InferenceClient(
    model='Salesforce/blip-image-captioning-large',
    token=settings.HUGGINGFACEHUB_API_TOKEN,
)

async def query_image_model(self, image_bytes):
    try:
        result = await asyncio.to_thread(_hf_client.image_to_text, image_bytes)
        return result.generated_text if hasattr(result, 'generated_text') else str(result)
    except Exception as exc:
        logger.warning("Image captioning API unavailable: %s", exc)
        return "Image uploaded successfully, but the image description service is currently unavailable."
```

**Key pattern:** `asyncio.to_thread()` runs the synchronous HuggingFace call in a thread pool, so it doesn't block the async event loop. The `try/except` ensures that if HuggingFace is down, the user gets a graceful fallback message instead of a WebSocket crash.

### Updating the image in the database:

```python
@staticmethod
def update_image_message(image_message, image_array, image_description):
    # Convert numpy array (BGR) to PIL Image (RGB)
    img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

    # Save to BytesIO → Django ContentFile
    img_io = io.BytesIO()
    img.save(img_io, format='PNG')
    image_content = ContentFile(img_io.getvalue(), name=f"image_{image_message.id}.png")

    # Update the database record
    image_message.image = image_content
    image_message.width, image_message.height = img.size
    image_message.description = image_description
    image_message.save()
```

**OpenCV color format note:** OpenCV uses BGR (Blue-Green-Red) order, while PIL and web browsers use RGB. `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` converts between them.

---

## 4. Voice Processing Pipeline

### Flow:

```
Base64 audio → Decode → Save as .wav file → Google STT → Transcript text
                                                         │
                                            "Hello, what is the weather?"
                                                         │
                                                         ▼
                                              LLM response → Optional TTS → Audio bytes
```

### VoiceModalHandler (`apps/general_assistant/services.py`):

```python
class VoiceModalHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    async def speech_to_text(self, audio_file):
        """Convert audio file to text using Google Speech Recognition."""
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
        """Convert text to audio using Google TTS."""
        def generate_audio():
            tts = gTTS(text=text, lang='en')
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            return temp_file.name

        audio_file = await asyncio.to_thread(generate_audio)
        with open(audio_file, 'rb') as f:
            audio_content = f.read()
        return audio_content

    async def get_audio_duration(self, audio_file):
        """Get duration of audio file using librosa."""
        def get_duration():
            y, sr = librosa.load(audio_file)
            return librosa.get_duration(y=y, sr=sr)
        return await asyncio.to_thread(get_duration)
```

### Audio format conversion:

```python
@staticmethod
async def convert_audio_format(input_file, output_file, target_sr=22050):
    """Resample audio to target sample rate."""
    def convert():
        y, sr = librosa.load(input_file, sr=None)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sf.write(output_file, y, target_sr)

    await asyncio.to_thread(convert)
```

**Why 22050 Hz?** This is a standard sample rate for speech processing. Higher rates waste computation; lower rates lose quality.

---

## 5. Database Models

### GeneralConversation:

```python
class GeneralConversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, default='Untitled Conversation')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

### GeneralMessage:

```python
class GeneralMessage(models.Model):
    CONTENT_TYPES = [('TE', 'Text'), ('IM', 'Image'), ('AU', 'Audio')]

    conversation = models.ForeignKey(GeneralConversation, on_delete=models.CASCADE)
    content_type = models.CharField(max_length=2, choices=CONTENT_TYPES)
    is_from_user = models.BooleanField(default=True)
    in_reply_to = models.ForeignKey('self', null=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)
```

### ImageMessage & AudioMessage:

```python
class ImageMessage(models.Model):
    message = models.OneToOneField(GeneralMessage, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='image_messages/')
    width = models.IntegerField(null=True)
    height = models.IntegerField(null=True)
    description = models.TextField(blank=True)

class AudioMessage(models.Model):
    message = models.OneToOneField(GeneralMessage, on_delete=models.CASCADE)
    audio_file = models.FileField(upload_to='voice_messages/')
    transcript = models.TextField(blank=True)
    duration = models.FloatField(null=True)
```

**OneToOneField** means each `GeneralMessage` can have exactly one `ImageMessage` OR one `AudioMessage` (or neither, for plain text).

---

## 6. Error Handling Strategy

External APIs (HuggingFace, Google Speech) can fail. The consumer wraps each handler in try/except:

```python
async def handle_image_message(self, content):
    try:
        image_data = base64.b64decode(content.get('content'))
        handler = ImageModalHandler()
        img_arr, description = await handler.process_image(image_data)
        # ... save and respond
    except Exception as exc:
        logger.error("Image processing failed: %s", exc)
        await self.send(text_data=json.dumps({
            'type': 'error',
            'message': 'Unable to process image. Please try again.'
        }))
```

**Principle:** Never let an external API failure crash the WebSocket connection. Always catch, log, and send a user-friendly error message.

---

## 7. The Complete asyncio.to_thread Pattern

Since most ML/audio libraries are synchronous, the project uses `asyncio.to_thread()` everywhere:

```python
# BAD — blocks the event loop (all other WebSocket connections freeze)
result = _hf_client.image_to_text(image_bytes)

# GOOD — runs in thread pool, event loop stays responsive
result = await asyncio.to_thread(_hf_client.image_to_text, image_bytes)
```

### Where it's used:

| Function | Why |
|----------|-----|
| `_hf_client.image_to_text()` | HuggingFace API call (blocking HTTP) |
| `_hf_client.text_generation()` | HuggingFace API call (blocking HTTP) |
| `recognizer.recognize_google()` | Google Speech API call |
| `gTTS().save()` | File I/O |
| `librosa.load()` | Audio file reading |
| `model.save()` | Database write |
| `BERTopic.load()` | Model loading from disk |

---

## 8. Tests

The test suite (`apps/general_assistant/tests/`) demonstrates how to mock external services:

```python
# test_services.py
@patch("general_assistant.services._hf_client")
async def test_success(self, mock_client):
    mock_result = MagicMock()
    mock_result.generated_text = "A dog running"
    mock_client.image_to_text.return_value = mock_result

    handler = ImageModalHandler()
    result = await handler.query_image_model(b"fake-bytes")

    assert result == "A dog running"

@patch("general_assistant.services._hf_client")
async def test_api_error_returns_fallback(self, mock_client):
    mock_client.image_to_text.side_effect = ConnectionError("DNS failed")

    handler = ImageModalHandler()
    result = await handler.query_image_model(b"fake-bytes")

    assert "unavailable" in result.lower()
```

**Testing principle:** Mock external APIs. Never make real API calls in tests.

---

## Exercises

1. **Add video message support** — Create a `VideoMessage` model, add a `video_message` handler to the consumer, and use a video captioning model to describe the video.
2. **Add audio format detection** — Use `python-magic` or file headers to detect the audio format and convert to WAV before processing.
3. **Add image OCR** — Use Tesseract or a HuggingFace OCR model to extract text from images.

---

## Key Files

| File | What It Does |
|------|-------------|
| `apps/general_assistant/consumers.py` | Multimodal WebSocket consumer |
| `apps/general_assistant/services.py` | ImageModalHandler + VoiceModalHandler |
| `apps/general_assistant/models.py` | GeneralConversation, GeneralMessage, ImageMessage, AudioMessage |
| `apps/general_assistant/tests/` | Test suite (25 tests) |
| `apps/convochat/utils/text_chat_handler.py` | Shared text response handler |
| `apps/convochat/utils/configure_llm.py` | LLM chain configuration |
