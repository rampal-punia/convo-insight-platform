# Intern Assignment 02: Fix the Celery NLP Pipeline — Broken Tasks & Missing Connections

**Track:** Backend — Celery + NLP Integration
**Difficulty:** Intermediate
**Estimated Effort:** 5–7 hours
**Prerequisites:** Assignment 01 completed, basic Celery understanding, read `docs/langgraph_workflow_vs_agents.md`

---

## Problem Statement

The `convochat` app has a Celery task pipeline designed to analyze conversations in the background — sentiment analysis, intent recognition, topic extraction, performance evaluation, and batch processing of recent messages.

**None of it works.**

The tasks call functions that don't exist, pass fields to models that don't have them, reference variables with typos, and the only Celery beat schedule entry points to a broken LLM fine-tuning task in a completely different app.

This means:
- Sentiment scores are never computed for conversations
- Intent predictions are never saved
- Topic distributions are never generated
- LLM performance is never evaluated
- The "process recent messages" batch task that should run every 30 minutes is completely commented out

---

## Root Cause Analysis

### Gap 1: Undefined Functions

**File:** `apps/convochat/tasks.py`

Three functions are called but **never defined or imported anywhere** in the entire codebase:

```python
# Line ~216 in process_user_message:
sentiment_result = perform_sentiment_analysis(user_message_content)   # ← undefined
intent_result = detect_intent(user_message_content)                   # ← undefined

# Line ~241 in process_ai_response:
topic_result = extract_topics(ai_response_content)                   # ← undefined
```

These should use the NLP utilities from `data_processing/preprocess.py` and the playground's `ModelManager`, or call the existing task functions (`analyze_sentiment`, `recognize_intent`, `analyze_topics`) that already exist in the same file.

### Gap 2: Wrong Model Fields in `save_user_message` and `save_ai_message`

**File:** `apps/convochat/tasks.py`

```python
# save_user_message (line ~103):
UserText.objects.create(
    conversation=conversation,  # ← UserText has no 'conversation' field
    content=content,            # ← correct
    is_from_user=True,          # ← UserText has no 'is_from_user' field
)
```

The actual `UserText` model (`apps/convochat/models.py`) has:
- `message` (OneToOneField to Message)
- `content` (TextField)
- `sentiment_score` (FloatField, nullable)
- `intent` (CharField, nullable)
- `primary_topic` (CharField, nullable)

Similarly, `save_ai_message` passes `is_from_user` and `in_reply_to` to `AIText.objects.create`, but `AIText` only has: `content`, `message`, `confidence_score`, `recommendation`, `tool_calls`.

### Gap 3: Variable Name Typo

**File:** `apps/convochat/tasks.py`, line ~27

```python
# analyze_conversation_topics task:
conversation = Conversation.objects.get(id=conversation_id)
# ... later:
for msg in Conversation.messages.all():   # ← Capital C = class, not instance
```

Should be `conversation.messages.all()` (lowercase `c` = the instance).

### Gap 4: Sentiment Category FK Mismatch

**File:** `apps/convochat/tasks.py`, `analyze_sentiment` task (~line 326)

```python
Sentiment.objects.create(
    category=result['category'],  # ← 'category' is a ForeignKey to SentimentCategory, not a string
    ...
)
```

The `Sentiment.category` field is a `ForeignKey(SentimentCategory)`, so you can't pass a raw string. You need to look up or create the `SentimentCategory` object first.

### Gap 5: Celery Beat Schedule Only Has One Broken Task

**File:** `config/celery.py`

```python
app.conf.beat_schedule = {
    'fine-tune-llm-weekly': {
        'task': 'apps.llms.tasks.run_fine_tuning',  # ← points to broken LLM task
        'schedule': crontab(day_of_week=6, hour=0, minute=0),
    },
}
```

The `process_recent_messages` task exists in `convochat/tasks.py` but its schedule is **commented out** (~lines 353-358). There are no periodic schedules for:
- Processing recent messages (every 30 min)
- Evaluating LLM performance (daily)
- Cleaning up stale conversations (weekly)

---

## Assignment Tasks

### Task 1: Fix `save_user_message` and `save_ai_message` (45 min)

**File:** `apps/convochat/tasks.py`

**1a.** Read the `UserText` and `AIText` models in `apps/convochat/models.py`. List all their fields.

**1b.** Rewrite `save_user_message` to match `UserText`'s actual fields:

```python
# The correct approach:
# 1. First create the Message object (which has conversation, content, is_from_user)
# 2. Then create the UserText linked to that Message

@shared_task
def save_user_message(conversation_id, content):
    conversation = Conversation.objects.get(id=conversation_id)
    message = Message.objects.create(
        conversation=conversation,
        content=content,
        is_from_user=True,
    )
    user_text = UserText.objects.create(
        message=message,
        content=content,
    )
    return str(user_text.id)
```

**1c.** Rewrite `save_ai_message` similarly to match `AIText`'s actual fields.

**Testing:** Create `convochat/tests/test_tasks.py` with tests that verify:
- `save_user_message` creates both a `Message` and a `UserText`
- `save_ai_message` creates both a `Message` and an `AIText`
- Invalid conversation_id raises `Conversation.DoesNotExist`

---

### Task 2: Fix the Undefined NLP Functions (60 min)

**File:** `apps/convochat/tasks.py`

**2a.** Decide on an approach for the three missing functions:

**Option A (Recommended):** Chain existing Celery tasks instead of calling undefined functions:
```python
@shared_task
def process_user_message(conversation_id, user_message_content):
    # Save the message first
    message_id = save_user_message.delay(conversation_id, user_message_content).get()

    # Chain analysis tasks
    analyze_sentiment.delay(message_id)
    recognize_intent.delay(message_id)

    return message_id
```

**Option B:** Create the missing functions as wrappers around the playground's `ModelManager` or the `data_processing` utilities.

**2b.** Implement your chosen approach. If using Option A, update `process_user_message` and `process_ai_response` to chain existing tasks.

**2c.** Fix the typo in `analyze_conversation_topics`: change `Conversation.messages.all()` to `conversation.messages.all()`.

---

### Task 3: Fix Sentiment Category FK Assignment (20 min)

**File:** `apps/convochat/tasks.py`, `analyze_sentiment` task

Update the `Sentiment.objects.create()` call to properly resolve the category:

```python
# Instead of:
Sentiment.objects.create(category=result['category'], ...)

# Do:
category_obj, _ = SentimentCategory.objects.get_or_create(
    name=result['category']
)
Sentiment.objects.create(
    category=category_obj,
    score=result['score'],
    ...
)
```

**Testing:** Add a test that verifies `analyze_sentiment` correctly creates a `Sentiment` with a `SentimentCategory` FK.

---

### Task 4: Fix and Expand the Celery Beat Schedule (30 min)

**File:** `config/celery.py`

**4a.** Remove the broken `fine-tune-llm-weekly` entry (the `apps.llms.tasks` module is broken — see Assignment 06).

**4b.** Add these periodic schedules:

```python
app.conf.beat_schedule = {
    'process-recent-messages': {
        'task': 'apps.convochat.tasks.process_recent_messages',
        'schedule': 1800,  # Every 30 minutes
    },
    'evaluate-llm-performance-daily': {
        'task': 'apps.convochat.tasks.evaluate_overall_performance',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    'cleanup-stale-conversations': {
        'task': 'apps.convochat.tasks.cleanup_stale_conversations',
        'schedule': crontab(day_of_week=0, hour=3, minute=0),  # Weekly Sunday 3 AM
    },
}
```

**4c.** Create the `cleanup_stale_conversations` task in `convochat/tasks.py`:

```python
@shared_task
def cleanup_stale_conversations():
    """Archive conversations inactive for > 24 hours."""
    from django.utils import timezone
    from datetime import timedelta

    threshold = timezone.now() - timedelta(hours=24)
    stale = Conversation.objects.filter(
        status=Conversation.Status.ACTIVE,
        modified__lt=threshold,
    )
    count = stale.update(status=Conversation.Status.ENDED)
    logger.info(f"Archived {count} stale conversations")
    return count
```

**Testing:** Write a test that creates an old active conversation, runs the cleanup task, and verifies it was archived.

---

### Task 5: Write Comprehensive Task Tests (45 min)

**File:** `apps/convochat/tests/test_tasks.py` (create)

Write tests for all fixed tasks using Celery's `task_always_eager` setting:

```python
import pytest
from django.test import override_settings

@pytest.fixture
def conversation(db, user):
    return Conversation.objects.create(user=user, title="Test")

@pytest.mark.django_db
@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class TestSaveUserMessage:
    def test_creates_message_and_user_text(self, conversation):
        from convochat.tasks import save_user_message
        result = save_user_message(str(conversation.id), "Hello")
        assert Message.objects.filter(conversation=conversation).exists()
        assert UserText.objects.filter(content="Hello").exists()
```

Cover:
- `save_user_message` / `save_ai_message` — happy path + error cases
- `cleanup_stale_conversations` — verifies stale conversations get archived
- `analyze_conversation_topics` — verifies it doesn't crash (mock the NLP model)

---

## File Reference Map

```
backend/
├── apps/
│   ├── convochat/
│   │   ├── tasks.py                      ← Tasks 1-5 (main file to fix)
│   │   ├── models.py                     ← Read-only (understand model fields)
│   │   └── tests/
│   │       └── test_tasks.py             ← Tasks 1, 5 (create)
│   └── analysis/
│       └── models.py                     ← Read-only (understand Sentiment model)
├── config/
│   └── celery.py                         ← Task 4 (fix beat schedule)
└── data_processing/
    └── preprocess.py                     ← Read-only (available NLP utilities)
```

---

## Key Concepts to Learn

1. **Celery task chaining** — Using `.delay()` and `.s()` to chain tasks asynchronously instead of calling sync functions
2. **Django model FK resolution** — Can't pass a string where a ForeignKey is expected; must resolve to an object first
3. **`CELERY_TASK_ALWAYS_EAGER`** — Setting that runs tasks synchronously in tests (no broker needed)
4. **Celery Beat** — The periodic task scheduler; `crontab()` for cron-like schedules, integers for seconds
5. **Python variable naming bugs** — `Conversation` (class) vs `conversation` (instance) — linters catch this but many don't

---

## Submission Checklist

- [ ] `save_user_message` creates `Message` + `UserText` with correct fields
- [ ] `save_ai_message` creates `Message` + `AIText` with correct fields
- [ ] `process_user_message` chains existing tasks (no undefined function calls)
- [ ] `process_ai_response` chains existing tasks (no undefined function calls)
- [ ] `analyze_conversation_topics` uses `conversation` (lowercase) not `Conversation`
- [ ] `analyze_sentiment` resolves `SentimentCategory` FK correctly
- [ ] Celery beat schedule has `process-recent-messages`, `evaluate-llm-performance-daily`, `cleanup-stale-conversations`
- [ ] `cleanup_stale_conversations` task works correctly
- [ ] All tests pass (`pytest apps/ -v`)
- [ ] New test file `convochat/tests/test_tasks.py` covers all fixed tasks

---

*Assignment created: May 2026*
*Series: ConvoInsight Platform — Intern Assignments*
