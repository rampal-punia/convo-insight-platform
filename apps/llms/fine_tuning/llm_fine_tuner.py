import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
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
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.output_dir, num_labels=num_labels)

    def prepare_dataset(self):
        # Load the dataset if csv file
        dataset = load_dataset('csv', data_files=self.dataset_path)

        # Split the datase
        train_dataset, test_dataset = train_test_split(
            dataset['train'], test_size=0.2)
        return train_dataset, test_dataset

    def preprocess_data(self, dataset, max_length=128):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['text'])
        tokenized_dataset = tokenized_dataset.rename_columns('label', 'labels')
        tokenized_dataset.set_format('torch')
        return tokenized_dataset

    def setup_trainer(self):
        train_dataset, test_dataset = self.prepare_dataset()

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

    # Training Loop
    def train_model(trainer, checkpoint_dir=None):
        if checkpoint_dir:
            return trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            return trainer.train()

    # Evaluation
    def evaluate_model(self, trainer):
        return trainer.evaluate()

    # Model saving and loading
    def save_model(self, trainer):
        trainer.save_model(self.output_dir)

    def load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.output_dir, num_labels=self.num_labels)
        tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        return model, tokenizer

    def fine_tune_model(self):
        # Prepare dataset
        train_dataset, test_dataset = self.prepare_dataset()

        # Preprocess dataset
        train_dataset = self.prepare_dataset(train_dataset)
        test_dataset = self.prepare_dataset(test_dataset)

        # Set up trainer
        trainer = self.setup_trainer(num_epochs=self.num_epochs)

        # Train model
        train_result = self.train_model(trainer)
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
        # model_name="mistralai/Mistral-7B-Instruct-v0.3",
        dataset_path="path/to/your/annotated/dataset",
        output_dir="fine_tuned_output",
    )
    fine_tuner.fine_tune_model()

    # To resume training from a checkpoint
    checkpoint_dir = "./fine_tuned_model/checkpoint-1000"
    trainer = fine_tuner.setup_trainer()
    # To load and use the fine-tuned model
    loaded_model, loaded_tokenizer = fine_tuner.load_model(
        model_name, num_labels, output_dir)
