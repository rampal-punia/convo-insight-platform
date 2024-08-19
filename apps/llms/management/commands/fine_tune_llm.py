# apps/llms/management/commands/fine_tune_llm.py

from django.core.management.base import BaseCommand
from fine_tuning.llm_fine_tuner import LLMFineTuner
from django.conf import settings

# run this command using
# python manage.py fine_tune_llm


class Command(BaseCommand):
    help = 'Fine-tune the LLM model on customer conversation datasets'

    def handle(self, *args, **options):
        dataset_paths = [
            "path/to/RSiCS/dataset",
            "path/to/3K_Conversations_Dataset",
            "path/to/Customer_Support_on_Twitter_Dataset"
        ]

        fine_tuner = LLMFineTuner(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            dataset_paths=dataset_paths,
            output_dir=settings.FINE_TUNED_MODEL_DIR
        )
        self.stdout.write(self.style.SUCCESS(
            'Starting fine-tuning process...'))
        fine_tuner.train()
        self.stdout.write(self.style.SUCCESS(
            'Fine-tuning completed successfully!'))
