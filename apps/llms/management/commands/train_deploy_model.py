# apps/llms/management/commands/train_deploy_model.py

from django.core.management.base import BaseCommand
from tasks import train_and_deploy_model


class Command(BaseCommand):
    help = 'Train and deploy a model using SageMaker'

    def add_arguments(self, parser):
        parser.add_argument(
            'model_type', type=str, help='Type of model to train (huggingface, pytorch, or sklearn)')
        parser.add_argument('script_path', type=str,
                            help='Path to the training script')
        parser.add_argument('train_data_path', type=str,
                            help='Path to the training data')
        parser.add_argument('output_path', type=str, help='S3 path for output')
        parser.add_argument('endpoint_name', type=str,
                            help='Name for the SageMaker endpoint')

    def handle(self, *args, **options):
        hyperparameters = {
            'epochs': 3,
            'learning_rate': 1e-5,
            # Add more hyperparameters as needed
        }

        train_and_deploy_model.delay(
            options['model_type'],
            options['script_path'],
            hyperparameters,
            options['train_data_path'],
            options['output_path'],
            options['endpoint_name']
        )

        self.stdout.write(self.style.SUCCESS(
            'Model training and deployment task has been queued'))
