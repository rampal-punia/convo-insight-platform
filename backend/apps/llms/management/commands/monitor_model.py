# apps/llms/management/commands/monitor_model.py

from django.core.management.base import BaseCommand
from tasks import monitor_model_performance


class Command(BaseCommand):
    help = 'Monitor the performance of a deployed model'

    def add_arguments(self, parser):
        parser.add_argument('endpoint_name', type=str,
                            help='Name of the SageMaker endpoint to monitor')

    def handle(self, *args, **options):
        monitor_model_performance.delay(options['endpoint_name'])
        self.stdout.write(self.style.SUCCESS(
            'Model monitoring task has been queued'))
