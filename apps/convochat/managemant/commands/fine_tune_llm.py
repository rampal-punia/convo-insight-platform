# convochat/management/commands/fine_tune_llm.py

from django.core.management.base import BaseCommand
from ml_model.fine_tuning.llm_fine_tuner import LLMFineTuner

# run this command using
# python manage.py fine_tune_llm


class Command(BaseCommand):
    help = 'Fine-tune the LLM model'

    def handle(self, *args, **options):
        fine_tuner = LLMFineTuner(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            dataset_path="path/to/your/annotated/dataset",
            output_dir="fine_tuned_output"
        )
        fine_tuner.train()
        self.stdout.write(self.style.SUCCESS(
            'Successfully fine-tuned the LLM model'))
