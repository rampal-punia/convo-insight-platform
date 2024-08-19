# config/celery.py

from celery import Celery
from celery.schedules import crontab
import os

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

app = Celery('config')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')


app.conf.beat_schedule = {
    'fine-tune-llm-weekly': {
        'task': 'apps.llms.tasks.run_fine_tuning',
        # Run every Sunday at midnight
        'schedule': crontab(day_of_week=6, hour=0, minute=0),
    },
}

# Set the start method for multiprocessing to 'spawn'
# try:
#     # Set the multiprocessing start method to 'spawn'
#     torch.multiprocessing.set_start_method('spawn', force=True)
# except RuntimeError:
#     pass
