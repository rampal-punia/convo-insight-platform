# run this command using
# python manage.py fine_tune_llm
from django.core.management.base import BaseCommand
from fine_tuning.llm_fine_tuner import LLMFineTuner
from django.conf import settings


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


# topic_model_tuner = LLMFineTuner(
#     model_name="bert-base-uncased", or BERTopic
#     dataset_path="path/to/topic_dataset.csv",
#     output_dir="fine_tuned_topic_model",
#     num_labels=10  # Adjust based on your number of topics
# )
# topic_model_tuner.fine_tune_model()

# intent_model_tuner = LLMFineTuner(
#     model_name="bert-base-uncased",
#     dataset_path="path/to/intent_dataset.csv",
#     output_dir="fine_tuned_intent_model",
#     num_labels=5  # Adjust based on your number of intents
# )
# intent_model_tuner.fine_tune_model()

# sentiment_model_tuner = LLMFineTuner(
#     model_name="roberta-base",
#     dataset_path="path/to/sentiment_dataset.csv",
#     output_dir="fine_tuned_sentiment_model",
#     num_labels=3  # For positive, negative, neutral
# )
# sentiment_model_tuner.fine_tune_model()
