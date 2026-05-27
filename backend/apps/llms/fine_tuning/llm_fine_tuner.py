import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split


class LLMFineTuner:
    def __init__(self, model_name, dataset_path, output_dir, num_labels=2, num_epochs=3, batch_size=8, learning_rate=2e-5):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=self.num_labels)

    def prepare_dataset(self):
        dataset = load_dataset('csv', data_files=self.dataset_path)
        train_dataset, test_dataset = train_test_split(
            dataset['train'], test_size=0.2)
        return train_dataset, test_dataset

    def preprocess_data(self, dataset, max_length=128):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['text'])
        tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
        tokenized_dataset.set_format('torch')
        return tokenized_dataset

    def setup_trainer(self, train_dataset, test_dataset):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=10,
            learning_rate=self.learning_rate,
            evaluation_strategy='steps',
            eval_steps=100,
            save_steps=100,
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        return trainer

    def train_model(self, trainer, checkpoint_dir=None):
        if checkpoint_dir:
            return trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            return trainer.train()

    def evaluate_model(self, trainer):
        return trainer.evaluate()

    def save_model(self, trainer):
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.output_dir, num_labels=self.num_labels)
        tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        return model, tokenizer

    def fine_tune_model(self, checkpoint_dir=None):
        # Prepare dataset
        train_dataset, test_dataset = self.prepare_dataset()

        # Preprocess dataset
        train_dataset = self.preprocess_data(train_dataset)
        test_dataset = self.preprocess_data(test_dataset)

        # Set up trainer
        trainer = self.setup_trainer(train_dataset, test_dataset)

        # Train model
        train_result = self.train_model(trainer, checkpoint_dir)
        print(f"Training results: {train_result}")

        # Evaluate model
        eval_result = self.evaluate_model(trainer)
        print(f"Evaluation results: {eval_result}")

        # Save model
        self.save_model(trainer)

        print(f"Model saved to {self.output_dir}")


# Usage example
if __name__ == "__main__":
    fine_tuner = LLMFineTuner(
        model_name="bert-base-uncased",
        dataset_path="path/to/your/dataset.csv",
        output_dir="fine_tuned_output",
    )
    fine_tuner.fine_tune_model()

    # To resume training from a checkpoint
    # fine_tuner.fine_tune_model(checkpoint_dir="./fine_tuned_output/checkpoint-1000")

    # To load and use the fine-tuned model
    loaded_model, loaded_tokenizer = fine_tuner.load_model()


# In management command use the below code:

# topic_model_tuner = LLMFineTuner(
#     model_name="bert-base-uncased",
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
