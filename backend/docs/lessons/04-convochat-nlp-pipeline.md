# Lesson 4: ConvoChat — NLP Pipeline & Background Tasks

> How Celery tasks, LangChain chains, sentiment analysis, and intent recognition power the chat backend.

---

## What You'll Learn

- Celery task queue for background processing
- LangChain chains (prompt → LLM → parser)
- Streaming LLM responses via WebSocket
- Sentiment analysis with fine-tuned BERT
- Intent recognition with transformer models
- Auto-title generation with HuggingFace InferenceClient

---

## 1. Celery Background Tasks

Some operations are too slow to run during a web request. Celery runs them in background workers.

### Setup:

```python
# config/celery.py
from celery import Celery

app = Celery('convo_insight')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
```

### How Celery works:

```
Django View/Consumer → task.delay(args) → Redis Queue → Celery Worker → task(args)
                                                                          │
                                                                          ▼
                                                                   Database / API
```

### Tasks in ConvoChat (`apps/convochat/tasks.py`):

```python
from celery import shared_task

@shared_task
def analyze_sentiment_task(message_id):
    """Background task: analyze sentiment of a message."""
    from convochat.utils.sentiment_analyzer import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    # ... analyze and save results

@shared_task
def recognize_intent_task(message_id):
    """Background task: classify user intent."""
    from convochat.utils.intent_recognizer_bertbase import IntentRecognizer
    recognizer = IntentRecognizer()
    # ... classify and save results

@shared_task
def generate_topic_summary_task(conversation_id):
    """Background task: generate topic summary."""
    # ... process and save
```

**Key pattern:** Heavy imports are inside the task function, not at module level. This keeps Celery worker startup fast.

### All ConvoChat Celery tasks:

| Task | What It Does |
|------|-------------|
| `analyze_sentiment_task` | Run BERT sentiment on a message |
| `recognize_intent_task` | Classify user intent |
| `generate_topic_summary_task` | Summarize conversation topics |
| `update_conversation_analytics` | Recalculate analytics for a conversation |
| `process_message_task` | Full NLP pipeline for a new message |
| `generate_conversation_title_task` | Auto-generate conversation title |

---

## 2. LangChain Chains — The Core Chat Engine

LangChain chains connect a **prompt template** → **LLM** → **output parser** into a pipeline.

### Architecture:

```
User Input + History
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  ChatPrompt  │────▶│  LLM Model   │────▶│ StrOutput    │
│  Template    │     │ (GPT-4o-mini │     │ Parser       │
│              │     │  or Mixtral) │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                     │
   Formats prompt     Generates tokens      Extracts text
   with variables     (streaming)           from response
```

### Code (`apps/convochat/utils/configure_llm.py`):

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

class LLMConfig:
    HUGGINGFACE_MODELS_REPO = {
        "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
        "Mixtral-8x7B-I": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        # ... more models
    }

    OPENAI_MODELS = {
        'gpt-4o-mini': 'gpt-4o-mini',
        "gpt-4": "gpt-4",
        # ... more models
    }

    @classmethod
    def configure_openai_llm(cls, model_name, **kwargs):
        return ChatOpenAI(
            model=cls.OPENAI_MODELS[model_name],
            openai_api_key=settings.OPENAI_API_KEY,
            **kwargs
        )

    @classmethod
    def configure_huggingface_llm(cls, model_name, **kwargs):
        return HuggingFaceEndpoint(
            repo_id=cls.HUGGINGFACE_MODELS_REPO[model_name],
            task='text-generation',
            huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
            **kwargs
        )
```

### Prompt Templates:

```python
class CustomPromptTemplates:
    @staticmethod
    def get_chat_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "Direct and efficient assistant. Provide complete, accurate responses."),
            ("human", "Context:\n{history}\n\nUser: {input}"),
            ("human", "Respond to the query while maintaining conversation context.")
        ])

    @staticmethod
    def get_doc_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "Extract relevant information from provided context."),
            ("human", "Context:\n{context}\n\nQuestion: {input}"),
            ("human", "Provide direct, evidence-based response.")
        ])
```

**Template variables:** `{history}`, `{input}`, `{context}` — these get filled in at runtime.

### Building the Chain:

```python
class ChainBuilder:
    @staticmethod
    def create_chat_chain(prompt, llm, run_name):
        output_parser = StrOutputParser()
        return prompt | llm | output_parser
        # LangChain uses the | (pipe) operator to chain components:
        # prompt → llm → parser
```

---

## 3. Streaming Responses to WebSocket

The chain's `astream_events()` method yields chunks as they arrive from the LLM:

```python
# apps/convochat/utils/text_chat_handler.py

class TextChatHandler:
    @staticmethod
    async def process_text_response(conversation, user_message, input_data, send_method):
        # Build the input with history
        history = await get_conversation_history(conversation.id)
        input_with_history = {
            'history': format_history(history),
            'input': input_data
        }

        # Stream the response
        llm_response_chunks = []
        async for chunk in configure_llm.main().astream_events(
            input_with_history,
            version='v2',
            include_names=['Assistant']
        ):
            if chunk['event'] in ['on_parser_start', 'on_parser_stream']:
                # Send each chunk to the frontend immediately
                await send_method(text_data=json.dumps(chunk))

            if chunk.get('event') == 'on_parser_end':
                output = chunk.get('data', {}).get('output', '')
                llm_response_chunks.append(output)

        # Join all chunks into complete response
        ai_response = ''.join(llm_response_chunks)

        # Save to database
        ai_message = await save_message(conversation, 'TE', is_from_user=False)
        await save_aitext(ai_message, ai_response)
```

### Event stream format:

```
{"event": "on_parser_start", ...}           ← streaming begins
{"event": "on_parser_stream", "data": {"chunk": "Python"}}    ← chunk 1
{"event": "on_parser_stream", "data": {"chunk": " is"}}       ← chunk 2
{"event": "on_parser_stream", "data": {"chunk": " great"}}    ← chunk 3
{"event": "on_parser_end", "data": {"output": "Python is great"}} ← complete
```

---

## 4. Auto-Title Generation

When a new conversation starts, the system auto-generates a title using HuggingFace:

```python
# apps/convochat/utils/configure_llm.py
from huggingface_hub import InferenceClient

_title_client = InferenceClient(
    model='czearing/article-title-generator',
    token=settings.HUGGINGFACEHUB_API_TOKEN,
)

async def generate_title(conversation_content):
    try:
        result = await asyncio.to_thread(
            _title_client.text_generation,
            conversation_content,
            max_new_tokens=50,
        )
        return result.strip() if result else 'Untitled Conversation'
    except Exception:
        return 'Untitled Conversation'
```

**`asyncio.to_thread()`** — runs the synchronous `InferenceClient` call in a thread pool, so it doesn't block the async event loop.

---

## 5. Sentiment Analysis

### Fine-tuned BERT model (`apps/convochat/utils/sentiment_analyzer.py`):

```python
class SentimentAnalyzer:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained('model_path')
        self.model = AutoModelForSequenceClassification.from_pretrained('model_path')

    def analyze_sentiment(self, text):
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probs).item()
        return sentiment  # 0=negative, 1=neutral, 2=positive
```

**Lazy import pattern:** The heavy `torch` and `transformers` imports are inside `__init__` and `analyze_sentiment`, not at module top level. This keeps Django/Celery startup fast.

---

## 6. Intent Recognition

### BERT-based intent classifier (`apps/convochat/utils/intent_recognizer_bertbase.py`):

```python
class IntentRecognizer:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained('model_path')
        self.model = AutoModelForSequenceClassification.from_pretrained('model_path')

    def recognize_intent(self, text):
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        return self.intent_labels[predicted_class]
```

### Intent labels:
- `order_status` — "Where is my order?"
- `product_inquiry` — "Tell me about this product"
- `complaint` — "This product is defective"
- `greeting` — "Hello, how are you?"
- etc.

---

## 7. Database Models

### Conversation & Messages (`apps/convochat/models.py`):

```python
class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, default='Untitled Conversation')
    created_at = models.DateTimeField(auto_now_add=True)

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    content_type = models.CharField(max_length=2, choices=CONTENT_TYPES)  # 'TE', 'IM', 'AU'
    is_from_user = models.BooleanField(default=True)
    in_reply_to = models.ForeignKey('self', null=True, on_delete=models.SET_NULL)

class AIText(models.Model):
    message = models.OneToOneField(Message, on_delete=models.CASCADE)
    content = models.TextField()
```

---

## 8. Complete NLP Pipeline for a Message

```
1. User sends message via WebSocket
   → Consumer receives JSON
   → Saves to database (Message + UserText)

2. Consumer dispatches Celery tasks:
   → analyze_sentiment_task.delay(message_id)
   → recognize_intent_task.delay(message_id)

3. Meanwhile, consumer starts streaming LLM response:
   → Load conversation history
   → Build LangChain chain (prompt | llm | parser)
   → astream_events() → yield chunks
   → Send each chunk to client via WebSocket
   → Save complete AI response to database

4. Celery workers process in background:
   → SentimentAnalyzer → save sentiment score
   → IntentRecognizer → save intent label
   → These don't block the chat response

5. If first message in conversation:
   → generate_title() via HuggingFace InferenceClient
   → Update conversation title in database
   → Send title update to client via WebSocket
```

---

## Exercises

1. **Add a new Celery task** — Create `summarize_conversation_task` that generates a summary using `generate_summary()` and saves it to the Conversation model.
2. **Add a new prompt template** — Create a "creative writing" prompt in `CustomPromptTemplates` with a different system message and chain it up.
3. **Add emotion detection** — Create a new Celery task that uses a pre-trained emotion model to detect emotions (joy, anger, fear, etc.) in messages.

---

## Key Files

| File | What It Does |
|------|-------------|
| `apps/convochat/tasks.py` | 17 Celery background tasks |
| `apps/convochat/models.py` | 9 models (Conversation, Message, AIText, etc.) |
| `apps/convochat/utils/configure_llm.py` | LLM config, prompt templates, chain builder |
| `apps/convochat/utils/text_chat_handler.py` | Streaming chat response handler |
| `apps/convochat/utils/sentiment_analyzer.py` | BERT sentiment analysis |
| `apps/convochat/utils/intent_recognizer_bertbase.py` | BERT intent recognition |
| `apps/convochat/utils/sentiment_2.py` | Alternative sentiment analyzer |
| `apps/convochat/consumers.py` | WebSocket consumer |
