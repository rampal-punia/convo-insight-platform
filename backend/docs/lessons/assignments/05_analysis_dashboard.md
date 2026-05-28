# Intern Assignment 05: Wire the Analysis Dashboard — Admin, Metrics & Periodic Evaluation

**Track:** Backend — Django Admin + Data Models + Celery
**Difficulty:** Intermediate
**Estimated Effort:** 4–5 hours
**Prerequisites:** Assignment 02 completed (Celery pipeline working), understand Django admin

---

## Problem Statement

The `analysis` app tracks LLM performance, conversation metrics, recommendations, topic distributions, and intent predictions. But the data is **barely visible**:

- **4 out of 5 models are NOT registered in the Django admin** — only `IntentPrediction` is visible
- **No admin customizations** — no filters, no search, no list_display, no readonly_fields
- **No periodic task to compute metrics** — `ConversationMetrics`, `LLMAgentPerformance`, and `Recommendation` records are never auto-generated
- **The `analysis/tasks.py` file is completely empty** — just a `# Create your tasks here.` comment

This means the admin panel shows almost nothing about how the AI agents are performing, what users are asking about, or how conversations are going.

---

## Root Cause Analysis

### What the Analysis Models Track

| Model | Purpose | Admin Registered? | Periodically Computed? |
|-------|---------|:--:|:--:|
| `LLMAgentPerformance` | Response time, accuracy, relevance per conversation | No | No |
| `ConversationMetrics` | Satisfaction score, avg response time per conversation | No | No |
| `Recommendation` | Improvement suggestions per conversation | No | No |
| `TopicDistribution` | Topic weights per conversation | No | Partial (Celery task exists) |
| `IntentPrediction` | Detected intent per user message | **Yes** | Partial (Celery task exists) |

### Current Admin State

**File:** `apps/analysis/admin.py`

```python
from django.contrib import admin
from .models import LLMAgentPerformance, IntentPrediction

@admin.register(IntentPrediction)
class IntentPredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'intent')
```

`LLMAgentPerformance` is imported but **never registered**. The other 3 models aren't even imported.

### Empty Tasks File

**File:** `apps/analysis/tasks.py`

```python
# Create your tasks here.
```

Completely empty — no periodic computation of metrics, no performance evaluation pipeline.

---

## Assignment Tasks

### Task 1: Register All Models in Admin (30 min)

**File:** `apps/analysis/admin.py`

Register all 5 models with rich admin configurations:

```python
from django.contrib import admin
from .models import (
    LLMAgentPerformance,
    ConversationMetrics,
    Recommendation,
    TopicDistribution,
    IntentPrediction,
)


@admin.register(LLMAgentPerformance)
class LLMAgentPerformanceAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'conversation', 'response_time',
        'accuracy_score', 'relevance_score',
        'overall_score', 'issue_resolved', 'evaluated_at',
    )
    list_filter = ('issue_resolved', 'evaluated_at')
    search_fields = ('conversation__id', 'feedback')
    readonly_fields = ('evaluated_at',)
    date_hierarchy = 'evaluated_at'
    ordering = ('-evaluated_at',)


@admin.register(ConversationMetrics)
class ConversationMetricsAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'conversation', 'overall_satisfaction_score',
        'average_response_time', 'sentiment_score', 'evaluated_at',
    )
    list_filter = ('evaluated_at',)
    search_fields = ('conversation__id',)
    readonly_fields = ('evaluated_at',)


@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'conversation', 'confidence_score',
        'is_applied', 'created_at',
    )
    list_filter = ('is_applied', 'created_at')
    search_fields = ('content', 'conversation__id')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)


@admin.register(TopicDistribution)
class TopicDistributionAdmin(admin.ModelAdmin):
    list_display = ('id', 'conversation', 'topic', 'weight')
    list_filter = ('topic',)
    search_fields = ('conversation__id',)


@admin.register(IntentPrediction)
class IntentPredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'intent', 'confidence_score', 'message')
    list_filter = ('intent',)
    search_fields = ('intent',)
```

**Verification:** Start the server, open `/admin/analysis/`, confirm all 5 models appear with their configured columns and filters.

---

### Task 2: Create Analysis Celery Tasks (60 min)

**File:** `apps/analysis/tasks.py`

Create periodic tasks that compute metrics for conversations that don't have them yet:

```python
"""Periodic analysis tasks for conversation metrics and LLM performance."""

import logging
from celery import shared_task
from django.utils import timezone

logger = logging.getLogger('analysis')


@shared_task
def compute_conversation_metrics():
    """Compute metrics for conversations that don't have metrics yet.

    Runs periodically. For each un-metrics conversation:
    1. Calculate average response time from message timestamps
    2. Average sentiment score from user messages
    3. Save ConversationMetrics record
    """
    from convochat.models import Conversation, UserText, AIText
    from .models import ConversationMetrics

    # Find active conversations without metrics
    conversations = Conversation.objects.filter(
        metrics__isnull=True,
        status=Conversation.Status.ACTIVE,
    ).select_related('metrics')[:50]  # Process in batches

    computed = 0
    for conv in conversations:
        try:
            # Average sentiment from UserText records
            user_texts = UserText.objects.filter(
                message__conversation=conv,
                sentiment_score__isnull=False,
            )
            avg_sentiment = (
                sum(ut.sentiment_score for ut in user_texts) / user_texts.count()
                if user_texts.exists() else None
            )

            # Average response time (time between user msg and AI reply)
            # ... calculate from message timestamps ...

            ConversationMetrics.objects.create(
                conversation=conv,
                sentiment_score=avg_sentiment,
                # average_response_time=...,
                # average_accuracy_score=...,
            )
            computed += 1
        except Exception as e:
            logger.error(f"Error computing metrics for {conv.id}: {e}")

    logger.info(f"Computed metrics for {computed} conversations")
    return computed


@shared_task
def evaluate_agent_performance():
    """Evaluate LLM performance for conversations with metrics but no performance record.

    For each conversation:
    1. Calculate response time from message pairs
    2. Generate placeholder accuracy/relevance scores
    3. Save LLMAgentPerformance record
    """
    from convochat.models import Conversation
    from .models import LLMAgentPerformance, ConversationMetrics

    # Find conversations with metrics but without performance evaluation
    conversations = Conversation.objects.filter(
        metrics__isnull=False,
        performance_evaluations__isnull=True,
    )[:50]

    evaluated = 0
    for conv in conversations:
        try:
            metrics = conv.metrics
            LLMAgentPerformance.objects.create(
                conversation=conv,
                response_time=metrics.average_response_time or timezone.timedelta(0),
                accuracy_score=metrics.average_accuracy_score,
                relevance_score=metrics.average_relevance_score,
                customer_satisfaction_score=metrics.overall_satisfaction_score,
                issue_resolved=conv.resolution_status == conv.ResolutionStatus.RESOLVED,
            )
            evaluated += 1
        except Exception as e:
            logger.error(f"Error evaluating {conv.id}: {e}")

    logger.info(f"Evaluated {evaluated} conversations")
    return evaluated


@shared_task
def generate_recommendations():
    """Generate improvement recommendations for low-scoring conversations.

    For conversations with satisfaction_score < 0.5 and no recommendations:
    1. Identify the issue (slow response? negative sentiment? unresolved?)
    2. Create a Recommendation record
    """
    from convochat.models import Conversation
    from .models import ConversationMetrics, Recommendation

    low_score_convs = Conversation.objects.filter(
        metrics__overall_satisfaction_score__lt=0.5,
        recommendations__isnull=True,
    )[:25]

    generated = 0
    for conv in low_score_convs:
        try:
            metrics = conv.metrics
            # Simple rule-based recommendations
            content = _generate_recommendation_content(conv, metrics)
            Recommendation.objects.create(
                conversation=conv,
                content=content,
                confidence_score=0.7,  # placeholder
            )
            generated += 1
        except Exception as e:
            logger.error(f"Error generating recommendation for {conv.id}: {e}")

    logger.info(f"Generated {generated} recommendations")
    return generated


def _generate_recommendation_content(conversation, metrics):
    """Generate a text recommendation based on conversation metrics."""
    issues = []
    if metrics.average_response_time and metrics.average_response_time.seconds > 300:
        issues.append("Response time exceeds 5 minutes — consider faster routing")
    if metrics.sentiment_score and metrics.sentiment_score < 0:
        issues.append("Negative user sentiment detected — review conversation quality")
    if conversation.resolution_status == conversation.ResolutionStatus.UNRESOLVED:
        issues.append("Conversation unresolved — escalate or follow up")
    return " | ".join(issues) if issues else "No specific issues identified"
```

---

### Task 3: Add Beat Schedule Entries (15 min)

**File:** `config/celery.py`

Add periodic schedules for the analysis tasks (in addition to what you added in Assignment 02):

```python
app.conf.beat_schedule = {
    # ... existing entries from Assignment 02 ...

    'compute-conversation-metrics': {
        'task': 'apps.analysis.tasks.compute_conversation_metrics',
        'schedule': crontab(minute=0, hour='*/2'),  # Every 2 hours
    },
    'evaluate-agent-performance': {
        'task': 'apps.analysis.tasks.evaluate_agent_performance',
        'schedule': crontab(minute=30, hour='*/6'),  # Every 6 hours
    },
    'generate-recommendations': {
        'task': 'apps.analysis.tasks.generate_recommendations',
        'schedule': crontab(minute=0, hour=3),  # Daily at 3 AM
    },
}
```

---

### Task 4: Write Tests (45 min)

**File:** `apps/analysis/tests/test_tasks.py` (create)

```python
import pytest
from django.test import override_settings
from django.utils import timezone
from datetime import timedelta

from convochat.models import Conversation, Message, UserText
from analysis.models import ConversationMetrics, LLMAgentPerformance, Recommendation


@pytest.fixture
def conversation(db, user):
    return Conversation.objects.create(user=user, title="Test Conv")


@pytest.mark.django_db
class TestComputeConversationMetrics:
    def test_creates_metrics_for_untracked_conversation(self, conversation):
        from analysis.tasks import compute_conversation_metrics

        # Create a UserText with a sentiment score
        msg = Message.objects.create(
            conversation=conversation, content="Hello", is_from_user=True
        )
        UserText.objects.create(
            message=msg, content="Hello", sentiment_score=0.8
        )

        result = compute_conversation_metrics()
        assert result >= 1
        assert ConversationMetrics.objects.filter(
            conversation=conversation
        ).exists()

    def test_skips_conversations_with_existing_metrics(self, conversation):
        from analysis.tasks import compute_conversation_metrics

        ConversationMetrics.objects.create(conversation=conversation)

        result = compute_conversation_metrics()
        # Should not create duplicate
        assert ConversationMetrics.objects.filter(
            conversation=conversation
        ).count() == 1


@pytest.mark.django_db
class TestEvaluateAgentPerformance:
    def test_creates_performance_for_metrics_conversations(self, conversation):
        from analysis.tasks import evaluate_agent_performance

        ConversationMetrics.objects.create(
            conversation=conversation,
            overall_satisfaction_score=0.7,
        )

        result = evaluate_agent_performance()
        assert result >= 1
        assert LLMAgentPerformance.objects.filter(
            conversation=conversation
        ).exists()


@pytest.mark.django_db
class TestGenerateRecommendations:
    def test_generates_for_low_score_conversations(self, conversation):
        from analysis.tasks import generate_recommendations

        ConversationMetrics.objects.create(
            conversation=conversation,
            overall_satisfaction_score=0.3,  # Below threshold
        )

        result = generate_recommendations()
        assert result >= 1
        rec = Recommendation.objects.get(conversation=conversation)
        assert rec.content  # Non-empty content
```

Cover:
- `compute_conversation_metrics` creates records and skips already-metrics conversations
- `evaluate_agent_performance` creates records and skips already-evaluated conversations
- `generate_recommendations` identifies low-score conversations and creates recommendations
- Empty conversation sets are handled gracefully (return 0)

---

## File Reference Map

```
backend/
├── apps/
│   ├── analysis/
│   │   ├── admin.py                  ← Task 1 (register all models)
│   │   ├── models.py                 ← Read-only (understand fields)
│   │   ├── tasks.py                  ← Task 2 (create from empty)
│   │   └── tests/
│   │       └── test_tasks.py         ← Task 4 (create)
│   └── convochat/
│       └── models.py                 ← Read-only (understand Conversation/UserText/AIText)
├── config/
│   └── celery.py                     ← Task 3 (add beat schedules)
└── docs/
    └── lessons/assignments/
        └── 05_analysis_dashboard.md  ← this file
```

---

## Key Concepts to Learn

1. **Django admin customization** — `list_display`, `list_filter`, `search_fields`, `date_hierarchy`, `readonly_fields`
2. **`select_related` vs `prefetch_related`** — Use `select_related` for ForeignKey (1:1), `prefetch_related` for reverse FK (1:many)
3. **Batch processing in Celery** — Process records in batches (`[:50]`) to avoid memory issues
4. **Rule-based recommendations** — Simple if/else rules based on metrics thresholds (no ML needed)
5. **Django `Q` objects and F expressions** — For complex queries like `metrics__isnull=True`

---

## Submission Checklist

- [ ] All 5 analysis models registered in admin with rich configurations
- [ ] `compute_conversation_metrics` task works and creates records
- [ ] `evaluate_agent_performance` task works and creates records
- [ ] `generate_recommendations` task works for low-score conversations
- [ ] Beat schedule has entries for all 3 new tasks
- [ ] Admin `/admin/analysis/` shows all 5 models with filters and search
- [ ] All existing tests pass
- [ ] New tests in `analysis/tests/test_tasks.py` pass

---

*Assignment created: May 2026*
*Series: ConvoInsight Platform — Intern Assignments*
