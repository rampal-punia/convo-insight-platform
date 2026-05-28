# Lesson 5: Playground — NLP Methods & Vector Search

> How the NLP playground provides 3 different text analysis methods and uses pgvector for semantic search.

---

## What You'll Learn

- Singleton pattern for model management
- Fine-tuned BERT for text classification
- Few-shot GPT prompting via OpenAI
- RAG (Retrieval-Augmented Generation) with pgvector
- Sentence embeddings and cosine similarity
- BERTopic for topic modeling
- Named Entity Recognition (NER)

---

## 1. The Playground Architecture

The NLP Playground is a WebSocket-based interface where users can test different NLP methods on text.

```
Frontend (NLP Playground UI)
       │
       ▼ WebSocket
NLPPlaygroundConsumer (consumers.py)
       │
       ├──► ModelManager (singleton)
       │       │
       │       ├── Sentiment Analysis
       │       ├── Intent Recognition
       │       ├── Topic Modeling (BERTopic)
       │       └── NER
       │
       ├──► Few-Shot GPT (OpenAI API)
       │
       └──► RAG Pipeline (pgvector + LangChain)
```

---

## 2. Singleton ModelManager

Loading ML models is expensive (takes seconds and lots of RAM). The `ModelManager` ensures each model is loaded **only once** and reused across requests.

### Thread-safe singleton pattern (`apps/playground/model_manager.py`):

```python
import threading

class ModelManager:
    _instance = None
    _lock = threading.Lock()
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:                    # Thread-safe check
                if cls._instance is None:       # Double-checked locking
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, model_name):
        if model_name not in self._models:
            self._models[model_name] = self._load_model(model_name)
        return self._models[model_name]

    def _load_model(self, model_name):
        if model_name == 'sentiment':
            from transformers import AutoModelForSequenceClassification
            return AutoModelForSequenceClassification.from_pretrained('...')
        # ... other models
```

**Why double-checked locking?**
- First check: Fast path — if instance exists, return immediately (no lock needed)
- `with cls._lock`: Only one thread can create the instance
- Second check: Another thread might have created it while we waited for the lock

---

## 3. Method 1: Fine-Tuned BERT Classification

A pre-trained BERT model fine-tuned on domain-specific data for text classification.

### How fine-tuned BERT works:

```
Original BERT (pretrained on all internet text)
       │
       ▼  Fine-tuning on labeled data
       │
Custom classifier head (num_labels = your categories)
       │
       ▼
Input: "I want to return my order"
Output: intent = "return_request" (confidence: 0.94)
```

### Implementation (`apps/playground/intent_recognitionwith_tr_bert.py`):

```python
class IntentModelTester:
    def __init__(self, model_path):
        import torch
        from transformers import AutoTokenizer
        from safetensors.torch import load_file

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load model weights from safetensors format
        state_dict = load_file(f"{model_path}/model.safetensors")
        # Build the model using a factory function (lazy import)
        model = _make_intent_classifier(num_labels=len(state_dict['classifier.weight']))
        model.load_state_dict(state_dict)
        self.model = model

    def predict(self, text):
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        return predicted_class, confidence
```

**Factory function pattern** — the `nn.Module` class is created inside a function to avoid importing `torch` at module level:

```python
def _make_intent_classifier(num_labels):
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class IntentClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        def forward(self, input_ids, attention_mask, **kwargs):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.dropout(outputs.pooler_output)
            return type('Output', (), {'logits': self.classifier(pooled)})()

    return IntentClassifier()
```

---

## 4. Method 2: Few-Shot GPT (OpenAI)

Instead of fine-tuning, we give GPT a few examples in the prompt. It learns from context.

### How few-shot works:

```
System: You are an intent classifier. Classify into: [greeting, complaint, order_status, ...]

User: "Hi there" → greeting
User: "Where is my package?" → order_status
User: "This is broken" → complaint

Now classify: "I need a refund"
→ return_request
```

### Implementation pattern:

```python
async def few_shot_classify(text, categories):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    examples = [
        ("Hello there!", "greeting"),
        ("Where is my order?", "order_status"),
        ("This product is defective", "complaint"),
        ("I want to return this", "return_request"),
    ]

    prompt = f"Classify this text into one of: {', '.join(categories)}\n\n"
    for ex_text, ex_label in examples:
        prompt += f'"{ex_text}" → {ex_label}\n'
    prompt += f'"{text}" →'

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
    )
    return response.choices[0].message.content.strip()
```

**Pros:** No training needed, adapts to new categories instantly
**Cons:** More expensive per call, slower, less consistent than fine-tuned models

---

## 5. Method 3: RAG with pgvector

Retrieval-Augmented Generation combines vector search with LLM generation.

### How RAG works:

```
User asks: "What is the return policy?"

Step 1: EMBED the query
  "What is the return policy?" → [0.023, -0.041, 0.078, ...]  (384-dim vector)

Step 2: SEARCH for similar documents
  SELECT content FROM documents
  ORDER BY embedding <-> query_vector  (cosine distance)
  LIMIT 5

Step 3: AUGMENT the prompt with retrieved context
  "Context: [retrieved doc 1] [retrieved doc 2] ..."
  "Question: What is the return policy?"

Step 4: GENERATE response using LLM
  GPT generates answer grounded in the retrieved documents
```

### pgvector Setup:

```sql
-- Enable the extension
CREATE EXTENSION vector;

-- Table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384),    -- 384-dimensional vector
    metadata JSONB
);

-- Create index for fast similarity search
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Embedding Model (`apps/playground/text_classification_vector_store.py`):

```python
class VectorStore:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings

    def embed_text(self, text):
        return self.model.encode(text).tolist()  # Returns 384-dim vector
```

### Django Model with vector field:

```python
from pgvector.django import VectorField

class Document(models.Model):
    content = models.TextField()
    embedding = VectorField(dimensions=384)
    metadata = models.JSONField(default=dict)

    def similarity_search(self, query_embedding, limit=5):
        return Document.objects.annotate(
            similarity=CosineDistance('embedding', query_embedding)
        ).order_by('similarity')[:limit]
```

---

## 6. Topic Modeling with BERTopic

BERTopic discovers topics in a collection of documents using BERT embeddings + clustering.

### How BERTopic works:

```
Documents → Sentence Embeddings → UMAP (dimensionality reduction)
    → HDBSCAN (clustering) → Topic clusters → c-TF-IDF (topic words)
```

### Implementation (`apps/playground/consumers.py`):

```python
async def _load_topic_model(self, model_path):
    from bertopic import BERTopic

    # Compatibility shims for loading models trained on older versions
    import torch.nn as nn
    if not hasattr(nn, 'BertSdpaSelfAttention'):
        nn.BertSdpaSelfAttention = nn.MultiheadAttention  # Shim for older models

    topic_model = await asyncio.to_thread(
        BERTopic.load, model_path
    )
    return topic_model

async def _get_topics(self, documents):
    topics, probs = self.topic_model.transform(documents)
    topic_info = self.topic_model.get_topic_info()
    return {
        'topics': topics.tolist(),
        'probabilities': probs.tolist(),
        'topic_info': topic_info.to_dict()
    }
```

---

## 7. Named Entity Recognition (NER)

Extracts entities (persons, organizations, dates, etc.) from text.

```python
# Uses a pre-trained token classification model
from transformers import pipeline

ner_pipeline = pipeline("ner", grouped_entities=True)
result = ner_pipeline("Apple was founded by Steve Jobs in California")

# Output:
# [{"entity_group": "ORG", "word": "Apple"},
#  {"entity_group": "PER", "word": "Steve Jobs"},
#  {"entity_group": "LOC", "word": "California"}]
```

---

## 8. Comparing the 3 Methods

| Aspect | Fine-tuned BERT | Few-Shot GPT | RAG + pgvector |
|--------|----------------|--------------|----------------|
| Training needed | Yes (hours) | No | No (just index docs) |
| Cost per call | Free (local) | ~$0.001 per call | Free (local) + optional LLM |
| Speed | ~50ms | ~500ms | ~100ms |
| Consistency | High | Medium | High (grounded in docs) |
| Best for | Fixed categories | Dynamic categories | Knowledge-heavy queries |

---

## Exercises

1. **Add a new classification category** — Add a new label to the fine-tuned BERT model. What steps are needed? (Hint: retrain with updated data)
2. **Implement hybrid search** — Combine pgvector similarity with keyword search (PostgreSQL `tsvector`) for better results.
3. **Add a new NER entity type** — Extend the NER pipeline to recognize product names or order IDs.

---

## Key Files

| File | What It Does |
|------|-------------|
| `apps/playground/consumers.py` | WebSocket consumer for NLP playground |
| `apps/playground/model_manager.py` | Singleton model loader |
| `apps/playground/intent_recognitionwith_tr_bert.py` | Fine-tuned BERT classifier |
| `apps/playground/text_classification_vector_store.py` | pgvector + sentence embeddings |
| `apps/playground/sentiment/` | Sentiment analysis models |
| `apps/playground/management/commands/` | Seed demo data, train models |
