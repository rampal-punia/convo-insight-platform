# Intern Assignment 04: Wire NLP Playground API — Few-Shot & RAG Methods

**Track:** Backend — NLP + REST API Integration
**Difficulty:** Intermediate-Advanced
**Estimated Effort:** 5–6 hours
**Prerequisites:** Assignment 01 completed, read `apps/playground/consumers.py` thoroughly

---

## Problem Statement

The NLP playground has a WebSocket consumer with **full implementations** for three analysis methods:

| Method | How It Works | WebSocket | REST API |
|--------|-------------|-----------|----------|
| `finetuned` | Local transformer models (BERT) | Works | **Works** |
| `few_shot_learning` | GPT with example prompts from DB | Works | **Returns 501** |
| `rag` | Vector store + GPT retrieval | Works | **Returns 501** |

The REST API (`/api/v1/nlp/sentiment/`, `/api/v1/nlp/intent/`, etc.) only supports the `finetuned` method. When users send `method: "few_shot_learning"` or `method: "rag"`, the API returns **HTTP 501 NOT IMPLEMENTED**.

The irony: the WebSocket consumer at `/ws/playground/` has complete, working implementations for all three methods sitting right there in `playground/consumers.py`. The REST API ViewSet just doesn't call them.

---

## Root Cause Analysis

**File:** `apps/api/v1/views_nlp.py`

Every action method in `NLPAnalysisViewSet` follows this pattern:

```python
@action(detail=False, methods=["post"])
def analyze_sentiment(self, request):
    method = request.data.get("method", "finetuned")

    if method == "finetuned":
        # ... working implementation using ModelManager
        return Response(output)
    else:
        return Response(
            {"error": f"Method '{method}' not implemented"},
            status=status.HTTP_501_NOT_IMPLEMENTED,
        )
```

Meanwhile, in `playground/consumers.py`, the `NLPPlaygroundConsumer` class has:

```python
async def analyze_sentiment(self, text, method):
    if method == "few_shot_learning":
        # Full implementation using ChatOpenAI with few-shot prompts
        # Loads examples from database, formats prompt, calls LLM
        ...
    elif method == "rag":
        # Full implementation using Redis vector store + retrieval
        # Queries similar documents, builds context, calls LLM
        ...
```

The same gap exists for all four NLP actions: `analyze_sentiment`, `analyze_intent`, `analyze_topic`, and `analyze_ner`.

The fix: extract the logic from the WebSocket consumer into reusable service functions, then call them from both the consumer and the ViewSet.

---

## Assignment Tasks

### Task 1: Study the WebSocket Consumer (30 min, reading only)

**File:** `apps/playground/consumers.py`

Read the entire file carefully. Understand:

1. **`ModelManager`** (lines ~24-257) — How it loads models, the singleton pattern, thread safety
2. **`EcommerceSentimentMapper`** (lines ~260-282) — How it maps base sentiments to e-commerce ones
3. **`NLPPlaygroundConsumer`** — For each of these method handlers, understand the full flow:
   - `analyze_sentiment` with `few_shot_learning` and `rag`
   - `analyze_intent` with `few_shot_learning` and `rag`
   - `analyze_topic` with `few_shot_learning` and `rag`

Answer these questions in a comment at the top of your service file (Task 2):
- What LLM does `few_shot_learning` use? Where is it initialized?
- What vector store does `rag` use? How are documents indexed?
- Where do the few-shot examples come from (database? hardcoded?)?
- What format does each method return its results in?

---

### Task 2: Create NLP Service Module (60 min)

**File:** `apps/playground/nlp_services.py` (create)

Extract the core analysis logic into **reusable async functions** that both the consumer and the ViewSet can call:

```python
"""
Reusable NLP analysis services.

Extracted from NLPPlaygroundConsumer so both WebSocket and REST API
can use the same analysis logic.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger('playground')


async def analyze_sentiment_few_shot(text: str, llm=None, cache=None) -> Dict[str, Any]:
    """Analyze sentiment using few-shot prompting with GPT.

    Args:
        text: The text to analyze
        llm: ChatOpenAI instance (caller should provide)
        cache: Redis-backed cache for prompt examples (optional)

    Returns:
        Dict with 'sentiment', 'confidence', 'method' keys
    """
    # Move the few_shot_learning logic from the consumer here
    # 1. Load examples from cache or database
    # 2. Format the few-shot prompt
    # 3. Call the LLM
    # 4. Parse and return the result
    ...


async def analyze_sentiment_rag(text: str, llm=None, vector_store=None) -> Dict[str, Any]:
    """Analyze sentiment using RAG (retrieval-augmented generation).

    Args:
        text: The text to analyze
        llm: ChatOpenAI instance
        vector_store: Redis vector store with indexed documents

    Returns:
        Dict with 'sentiment', 'confidence', 'method', 'sources' keys
    """
    # Move the rag logic from the consumer here
    ...
```

Create similar functions for intent analysis and topic analysis:
- `analyze_intent_few_shot(text, ...)`
- `analyze_intent_rag(text, ...)`
- `analyze_topic_few_shot(text, ...)`
- `analyze_topic_rag(text, ...)`

**Important:** These should be plain async functions, NOT methods on a class. This makes them easy to call from both the consumer and the ViewSet.

---

### Task 3: Update the REST API ViewSet (45 min)

**File:** `apps/api/v1/views_nlp.py`

Wire the `few_shot_learning` and `rag` methods into each action:

```python
from playground.nlp_services import (
    analyze_sentiment_few_shot,
    analyze_sentiment_rag,
    analyze_intent_few_shot,
    analyze_intent_rag,
    analyze_topic_few_shot,
    analyze_topic_rag,
)


class NLPAnalysisViewSet(viewsets.ViewSet):
    # ... existing code ...

    @action(detail=False, methods=["post"])
    async def analyze_sentiment(self, request):
        text = request.data.get("text", "")
        method = request.data.get("method", "finetuned")

        if method == "finetuned":
            # ... existing finetuned code ...
            return Response(output)

        elif method == "few_shot_learning":
            result = await analyze_sentiment_few_shot(text)
            return Response(result)

        elif method == "rag":
            result = await analyze_sentiment_rag(text)
            return Response(result)

        else:
            return Response(
                {"error": f"Unknown method: {method}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
```

Repeat for `analyze_intent` and `analyze_topic` actions.

**Note:** Since these are now async functions, you may need to make the ViewSet actions `async def` or use `asyncio.run()`. DRF supports async views in recent versions — check if your version does.

---

### Task 4: Refactor the WebSocket Consumer (30 min)

**File:** `apps/playground/consumers.py`

Update the consumer's analysis methods to call the new service functions instead of duplicating logic:

```python
# Before (consumer has all the logic inline):
async def analyze_sentiment(self, text, method):
    if method == "few_shot_learning":
        # 40 lines of inline logic...

# After (consumer delegates to service):
from .nlp_services import (
    analyze_sentiment_few_shot,
    analyze_sentiment_rag,
    # ...
)

async def analyze_sentiment(self, text, method):
    if method == "few_shot_learning":
        return await analyze_sentiment_few_shot(
            text, llm=self.llm, cache=self.cache
        )
    elif method == "rag":
        return await analyze_sentiment_rag(
            text, llm=self.llm, vector_store=self.vector_store
        )
```

This ensures the consumer and REST API use **exactly the same** analysis logic — no divergence.

---

### Task 5: Write Tests (45 min)

**File:** `apps/playground/tests/test_nlp_services.py` (create)

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_analyze_sentiment_few_shot():
    """Verify few-shot sentiment returns expected structure."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AsyncMock(
        content='{"sentiment": "positive", "confidence": 0.92}'
    )

    from playground.nlp_services import analyze_sentiment_few_shot
    result = await analyze_sentiment_few_shot("Great product!", llm=mock_llm)

    assert result["method"] == "few_shot_learning"
    assert "sentiment" in result
    assert "confidence" in result


@pytest.mark.asyncio
async def test_analyze_sentiment_invalid_method():
    """REST API should return 400 for unknown methods."""
    from rest_framework.test import APIRequestFactory
    from api.v1.views_nlp import NLPAnalysisViewSet

    factory = APIRequestFactory()
    request = factory.post("/api/v1/nlp/sentiment/", {
        "text": "Hello",
        "method": "nonexistent",
    })
    view = NLPAnalysisViewSet.as_view({"post": "analyze_sentiment"})
    response = view(request)

    assert response.status_code == 400
```

Cover:
- Each service function returns the expected structure
- REST API returns 200 for `few_shot_learning` and `rag` methods
- REST API returns 400 for unknown methods (not 501)
- REST API returns 400 for missing text
- WebSocket consumer still works with all three methods

---

## File Reference Map

```
backend/
├── apps/
│   ├── playground/
│   │   ├── consumers.py                  ← Task 1 (read), Task 4 (refactor)
│   │   ├── nlp_services.py               ← Task 2 (create — the main work)
│   │   └── tests/
│   │       └── test_nlp_services.py      ← Task 5 (create)
│   └── api/v1/
│       └── views_nlp.py                  ← Task 3 (wire few-shot + rag)
└── docs/
    └── lessons/assignments/
        └── 04_nlp_playground_api.md      ← this file
```

---

## Key Concepts to Learn

1. **Service Layer Pattern** — Extract business logic out of consumers/views into reusable service functions
2. **Few-Shot Prompting** — Give the LLM 3-5 labeled examples in the prompt so it learns the task format
3. **RAG (Retrieval-Augmented Generation)** — Query a vector store for relevant documents, inject them into the LLM prompt as context
4. **DRF async views** — DRF 3.14+ supports `async def` actions; earlier versions need `asyncio.run()` wrappers
5. **WebSocket vs REST tradeoffs** — WebSocket for streaming/real-time, REST for stateless queries

---

## Submission Checklist

- [ ] `nlp_services.py` created with extracted analysis functions
- [ ] `analyze_sentiment_few_shot` and `analyze_sentiment_rag` work correctly
- [ ] `analyze_intent_few_shot` and `analyze_intent_rag` work correctly
- [ ] `analyze_topic_few_shot` and `analyze_topic_rag` work correctly
- [ ] REST API returns 200 for `few_shot_learning` method (not 501)
- [ ] REST API returns 200 for `rag` method (not 501)
- [ ] REST API returns 400 for unknown methods (not 501)
- [ ] WebSocket consumer refactored to use service functions
- [ ] All existing tests pass
- [ ] New tests cover service functions and API endpoints

---

*Assignment created: May 2026*
*Series: ConvoInsight Platform — Intern Assignments*
