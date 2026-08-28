import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

app = Celery('vision')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Set up Celery Beat schedule for continuous learning
from celery.schedules import crontab
app.conf.beat_schedule = {
    'run-daily-learning': {
        'task': 'learning.run_daily_learning',
        # Default 02:00 AM UTC. This should be restarted if changed in DB.
        'schedule': crontab(hour=2, minute=0),
    },
}
