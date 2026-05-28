# Lesson 8: Analysis, LLMs & Advanced Topics

> How the platform evaluates agent performance, fine-tunes LLMs, and handles deployment concerns.

---

## What You'll Learn

- Agent performance evaluation metrics
- LLM fine-tuning pipeline
- SageMaker integration for model training
- Management commands for operational tasks
- Configuration and deployment patterns
- The sys.path import trick used in this project

---

## 1. Analysis — Agent Performance Evaluation

The `analysis` app evaluates how well the support agent performs across conversations.

### Models (`apps/analysis/models.py`):

```python
class AgentPerformance(models.Model):
    agent = models.ForeignKey(User, on_delete=models.CASCADE)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    total_messages = models.IntegerField(default=0)
    avg_response_time = models.FloatField(default=0.0)
    customer_satisfaction = models.FloatField(null=True)  # 0-5 rating
    resolution_status = models.CharField(choices=[
        ('resolved', 'Resolved'),
        ('escalated', 'Escalated'),
        ('unresolved', 'Unresolved'),
    ])
    created_at = models.DateTimeField(auto_now_add=True)


class ConversationMetrics(models.Model):
    conversation = models.OneToOneField(Conversation, on_delete=models.CASCADE)
    sentiment_score = models.FloatField(default=0.0)      # -1 to 1
    intent_accuracy = models.FloatField(default=0.0)       # 0 to 1
    avg_confidence = models.FloatField(default=0.0)        # 0 to 1
    topic_diversity = models.FloatField(default=0.0)       # 0 to 1
```

### AgentPerformanceEvaluator (`apps/analysis/agent_performance_evaluator.py`):

```python
class AgentPerformanceEvaluator:
    async def evaluate_conversation(self, conversation_id):
        messages = await self.get_messages(conversation_id)

        metrics = {
            'response_time': self._calc_avg_response_time(messages),
            'resolution_rate': self._calc_resolution_rate(messages),
            'sentiment_trend': await self._analyze_sentiment_trend(messages),
            'intent_accuracy': self._calc_intent_accuracy(messages),
            'topic_coherence': self._calc_topic_coherence(messages),
        }

        return metrics

    def _calc_avg_response_time(self, messages):
        """Average time between user message and agent response."""
        response_times = []
        for i in range(1, len(messages)):
            if messages[i].is_from_user and not messages[i-1].is_from_user:
                continue
            if not messages[i].is_from_user and messages[i-1].is_from_user:
                delta = (messages[i].created_at - messages[i-1].created_at).total_seconds()
                response_times.append(delta)
        return sum(response_times) / len(response_times) if response_times else 0

    def _calc_resolution_rate(self, messages):
        """Did the conversation end with a resolution?"""
        last_user_msg = [m for m in messages if m.is_from_user][-1]
        resolution_keywords = ['thanks', 'resolved', 'works now', 'great']
        return any(kw in last_user_msg.content.lower() for kw in resolution_keywords)
```

### Metrics explained:

| Metric | What It Measures | Good Score |
|--------|-----------------|------------|
| `avg_response_time` | Seconds between user message and AI response | < 2 seconds |
| `resolution_rate` | % of conversations ending in resolution | > 80% |
| `sentiment_trend` | Whether sentiment improves over the conversation | Positive trend |
| `intent_accuracy` | How often the intent classifier is correct | > 90% |
| `topic_coherence` | How relevant responses are to the conversation topic | > 0.7 |

---

## 2. LLM Fine-Tuning Pipeline

The `llms` app provides a pipeline for fine-tuning language models on custom data.

### LLMFineTuner (`apps/llms/fine_tuning/llm_fine_tuner.py`):

```python
class LLMFineTuner:
    def __init__(self, base_model, output_dir):
        self.base_model = base_model
        self.output_dir = output_dir

    def prepare_dataset(self, data_path):
        """Load and format training data."""
        from datasets import load_dataset
        dataset = load_dataset('json', data_files=data_path)
        return dataset

    def tokenize_dataset(self, dataset, tokenizer):
        """Tokenize the dataset for training."""
        def tokenize_fn(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512
            )
        return dataset.map(tokenize_fn, batched=True)

    def train(self, dataset, epochs=3, batch_size=8):
        """Fine-tune the model."""
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            TrainingArguments,
            Trainer,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForCausalLM.from_pretrained(self.base_model)

        tokenized = self.tokenize_dataset(dataset, tokenizer)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=2e-5,
            weight_decay=0.01,
            save_strategy='epoch',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized['train'],
            tokenizer=tokenizer,
        )

        trainer.train()
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
```

### Fine-tuning data format:

```json
{"text": "User: What is the return policy?\nAgent: Our return policy allows returns within 30 days..."}
{"text": "User: I want to cancel order 123\nAgent: I'll help you cancel order 123. Let me look that up..."}
```

### The fine-tuning workflow:

```
1. Collect conversation data (from production database)
2. Clean and format as JSONL (prompt-response pairs)
3. Split into train/validation sets
4. Tokenize with the base model's tokenizer
5. Train for N epochs with small learning rate (2e-5)
6. Evaluate on validation set
7. Save the fine-tuned model
8. Deploy (replace base model in ModelManager)
```

---

## 3. SageMaker Integration

For large-scale training, the project can use AWS SageMaker:

```python
# apps/llms/fine_tuning/llm_fine_tuner.py
class SageMakerTrainer:
    def __init__(self, role_arn, instance_type='ml.g4dn.xlarge'):
        self.role = role_arn
        self.instance_type = instance_type

    def launch_training_job(self, input_data_s3, output_s3, hyperparameters):
        """Launch a SageMaker training job."""
        from sagemaker.huggingface import HuggingFace

        estimator = HuggingFace(
            entry_point='train.py',
            source_dir='src/',
            instance_type=self.instance_type,
            instance_count=1,
            role=self.role,
            transformers_version='4.37',
            pytorch_version='2.1',
            py_version='py310',
            hyperparameters=hyperparameters,
        )

        estimator.fit({'train': input_data_s3})
        return estimator.latest_training_job.name
```

**When to use SageMaker vs local:**
- **Local**: Small models (< 1B params), small datasets, development
- **SageMaker**: Large models (7B+ params), large datasets, production training

---

## 4. Management Commands

Django management commands are CLI tools for operational tasks.

### Structure:

```
apps/
├── accounts/management/commands/create_demo_users.py
├── orders/management/commands/seed_orders.py
├── convochat/management/commands/
│   ├── seed_conversations.py
│   └── train_intent_model.py
├── playground/management/commands/
│   └── seed_playground_data.py
└── dashboard/management/commands/
    └── seed_demo.py
```

### Example: `create_demo_users.py`

```python
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User

class Command(BaseCommand):
    help = 'Create demo users for testing'

    def add_arguments(self, parser):
        parser.add_argument('--count', type=int, default=5)

    def handle(self, *args, **options):
        count = options['count']
        for i in range(count):
            user, created = User.objects.get_or_create(
                username=f'demo_user_{i}',
                defaults={'email': f'demo{i}@example.com'}
            )
            if created:
                user.set_password('demo123')
                user.save()
                self.stdout.write(f'Created user: {user.username}')
        self.stdout.write(self.style.SUCCESS(f'Created {count} demo users'))
```

### Running management commands:

```bash
python manage.py create_demo_users --count 10
python manage.py seed_orders
python manage.py seed_demo              # Seeds all demo data
python manage.py train_intent_model     # Trains the intent classifier
```

---

## 5. The sys.path Import Trick

This project uses a clever `sys.path` modification to enable clean imports:

```python
# config/settings/base.py
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "apps"))
```

This allows importing apps directly by name:

```python
# Instead of:
from apps.products.models import Product

# You can write:
from products.models import Product
```

**Why this matters:**
- Cleaner imports throughout the codebase
- Apps are more portable (could be extracted to separate packages)
- Matches Django's convention of importing by app name

---

## 6. Configuration Patterns

### Settings structure:

```
config/settings/
├── base.py       # All settings (used for development)
├── dev.py        # Development overrides
└── prod.py       # Production overrides
```

### Key settings for this project:

```python
# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'convo_insight',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

# Redis (for Celery + Channels)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://localhost:6379/0',
    }
}

# Celery
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

# Channels
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {"hosts": [('127.0.0.1', 6379)]},
    },
}

# API Keys (from environment variables)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
```

---

## 7. Abstract Base Model

A common pattern in this project — an abstract model that all models inherit from:

```python
# config/settings/base.py or a common models file
class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True
```

Any model that inherits from `TimeStampedModel` automatically gets `created_at` and `updated_at` fields.

---

## 8. Deployment Checklist

When deploying this project to production:

### Environment variables:

```bash
DJANGO_SETTINGS_MODULE=config.settings.prod
SECRET_KEY=<your-secret-key>
OPENAI_API_KEY=<your-openai-key>
HUGGINGFACEHUB_API_TOKEN=<your-hf-token>
DATABASE_URL=postgres://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0
```

### Services needed:

| Service | Purpose | Port |
|---------|---------|------|
| PostgreSQL 17 | Database + pgvector | 5432 |
| Redis 7 | Cache + Celery broker + Channel layer | 6379 |
| Daphne | ASGI server (HTTP + WebSocket) | 8000 |
| Celery worker | Background tasks | — |
| Celery beat | Scheduled tasks | — |

### Running in production:

```bash
# Start ASGI server
daphne config.asgi:application --port 8000 --bind 0.0.0.0

# Start Celery workers
celery -A config worker --loglevel=info --concurrency=4

# Start Celery beat (scheduler)
celery -A config beat --loglevel=info
```

---

## 9. Project-Wide Patterns Summary

### Lazy import pattern (used everywhere):
```python
# BAD — slow startup
import torch
from transformers import AutoModel

# GOOD — fast startup, load only when needed
def analyze(text):
    import torch
    from transformers import AutoModel
    ...
```

### asyncio.to_thread pattern (for sync code in async context):
```python
# BAD — blocks event loop
result = sync_function(args)

# GOOD — runs in thread pool
result = await asyncio.to_thread(sync_function, args)
```

### Error handling for external APIs:
```python
try:
    result = await api_call()
except (ConnectionError, TimeoutError) as exc:
    logger.warning("API unavailable: %s", exc)
    return fallback_response
```

---

## Exercises

1. **Create an evaluation dashboard** — Build a management command that generates a weekly performance report for all agents, aggregating metrics from the `analysis` app.
2. **Add A/B testing for prompts** — Create a model that stores different prompt variants and tracks which one performs better over time.
3. **Add model versioning** — Track which model version was used for each prediction, so you can compare performance across versions.

---

## Key Files

| File | What It Does |
|------|-------------|
| `apps/analysis/models.py` | Performance and metrics models |
| `apps/analysis/agent_performance_evaluator.py` | Evaluation logic |
| `apps/llms/fine_tuning/llm_fine_tuner.py` | Fine-tuning pipeline |
| `apps/dashboard/views.py` | Health check endpoints |
| `apps/dashboard/management/commands/seed_demo.py` | Demo data seeder |
| `config/settings/base.py` | All configuration |
| `config/asgi.py` | ASGI application |
| `config/celery.py` | Celery configuration |
