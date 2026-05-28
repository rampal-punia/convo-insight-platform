# Quick Win 08: Celery — Background Tasks

> Slow operations (ML inference, email, reports) run in background workers, not blocking web requests.

---

## The Problem Celery Solves

Imagine running BERT sentiment analysis during a chat request:

```
Without Celery:
  User sends message → Server runs BERT (3 seconds) → Server responds
  User waits 3 seconds staring at a blank screen

With Celery:
  User sends message → Server responds immediately
  Background: Celery worker runs BERT → saves result to database
  User experience: instant response, analysis appears moments later
```

---

## Architecture

```
Django (web process)
    │
    │ task.delay(args)           # "Fire and forget"
    │
    ▼
Redis (message broker)
    │
    │ queue: celery
    │
    ▼
Celery Worker
    │
    │ task(args)                 # Actually runs the function
    │
    ▼
Database / External API
```

- **Broker** (Redis) — holds tasks in a queue until a worker picks them up
- **Worker** — a separate process that executes tasks
- **Result backend** (Redis) — stores task results/status

---

## Defining Tasks

```python
# apps/convochat/tasks.py
from celery import shared_task

@shared_task
def analyze_sentiment_task(message_id):
    """Analyze sentiment of a message in the background."""
    # Lazy imports — keep worker startup fast
    from convochat.utils.sentiment_analyzer import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    message = Message.objects.get(id=message_id)
    sentiment = analyzer.analyze_sentiment(message.user_text.content)

    message.sentiment_score = sentiment
    message.save()

    return sentiment  # Stored in result backend


@shared_task
def recognize_intent_task(message_id):
    """Classify user intent in the background."""
    from convochat.utils.intent_recognizer_bertbase import IntentRecognizer

    recognizer = IntentRecognizer()
    message = Message.objects.get(id=message_id)
    intent = recognizer.recognize_intent(message.user_text.content)

    message.intent_label = intent
    message.save()

    return intent
```

**Key pattern:** Heavy imports (`transformers`, `torch`) are **inside** the task function, not at module level. This keeps the Celery worker startup fast (seconds, not minutes).

---

## Calling Tasks

```python
# From a view or consumer:
from convochat.tasks import analyze_sentiment_task

# Option 1: Fire and forget (most common)
analyze_sentiment_task.delay(message_id)

# Option 2: Schedule for later
analyze_sentiment_task.apply_async(args=[message_id], countdown=60)  # Run after 60 seconds

# Option 3: Get the result (avoid in web requests — defeats the purpose)
result = analyze_sentiment_task.delay(message_id)
result.get()  # Blocks until task completes
```

---

## Chaining Tasks

Run tasks in sequence:

```python
from celery import chain

# Run analyze_sentiment → then recognize_intent → then update_analytics
workflow = chain(
    analyze_sentiment_task.s(message_id),
    recognize_intent_task.s(),
    update_conversation_analytics.s(conversation_id),
)
workflow.apply_async()
```

---

## Periodic Tasks (Celery Beat)

Run tasks on a schedule:

```python
# config/celery.py
from celery.schedules import crontab

app.conf.beat_schedule = {
    'cleanup-expired-sessions': {
        'task': 'accounts.tasks.cleanup_sessions',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    'generate-daily-report': {
        'task': 'analysis.tasks.daily_report',
        'schedule': crontab(hour=8, minute=0),  # Daily at 8 AM
    },
}
```

---

## Running Celery

```bash
# Start the worker (in a separate terminal)
celery -A config worker --loglevel=info

# Start the beat scheduler (for periodic tasks)
celery -A config beat --loglevel=info

# Monitor tasks in real-time (requires flower)
celery -A config flower
# → Open http://localhost:5555
```

---

## Task States

```
PENDING → STARTED → SUCCESS
                  → FAILURE
                  → RETRY → SUCCESS
                         → FAILURE
```

```python
from celery.result import AsyncResult

result = AsyncResult('task-uuid')
result.status    # → 'PENDING', 'STARTED', 'SUCCESS', 'FAILURE'
result.result    # → The return value (if SUCCESS)
result.traceback # → Error traceback (if FAILURE)
```

---

## Error Handling in Tasks

```python
@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def call_external_api(self, url):
    try:
        response = requests.get(url)
        return response.json()
    except requests.RequestException as exc:
        # Retry up to 3 times with 60-second delay
        raise self.retry(exc=exc)
```

- `bind=True` — gives access to `self` (the task instance)
- `max_retries=3` — retry up to 3 times before giving up
- `default_retry_delay=60` — wait 60 seconds between retries

---

## Tasks in This Project

| App | Tasks | Purpose |
|-----|-------|---------|
| `convochat` | 17 tasks | Sentiment, intent, topics, summaries, titles |
| `orders` | 2 tasks | Intent recognition, sentiment for orders |
| `llms` | 3 tasks | Fine-tuning, monitoring, deployment |
| `analysis` | Multiple | Performance evaluation, metrics |

---

## Quick Exercise

1. Read `apps/convochat/tasks.py` — see how tasks are defined with lazy imports
2. Start a Celery worker: `celery -A config worker --loglevel=info`
3. In the Django shell, call a task:
   ```python
   from convochat.tasks import analyze_sentiment_task
   result = analyze_sentiment_task.delay(1)
   result.status   # Check status
   result.result   # Check result
   ```
4. Write a new task that sends a "welcome" notification when a user signs up
