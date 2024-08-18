# ml_model/fine_tuning/llm_fine_tuner.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import os


class LLMFineTuner:
    def __init__(self, model_name, dataset_path, output_dir):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def prepare_dataset(self):
        dataset = load_dataset(self.dataset_path)

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        return tokenized_datasets

    def train(self, num_epochs=3, batch_size=8, learning_rate=2e-5):
        tokenized_datasets = self.prepare_dataset()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=10,
            learning_rate=learning_rate,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
        )

        trainer.train()
        self.model.save_pretrained(os.path.join(
            self.output_dir, "fine_tuned_model"))
        self.tokenizer.save_pretrained(
            os.path.join(self.output_dir, "fine_tuned_model"))

    def load_fine_tuned_model(self):
        fine_tuned_path = os.path.join(self.output_dir, "fine_tuned_model")
        self.model = AutoModelForCausalLM.from_pretrained(fine_tuned_path)
        self.tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)


# Usage example
if __name__ == "__main__":
    fine_tuner = LLMFineTuner(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        dataset_path="path/to/your/annotated/dataset",
        output_dir="fine_tuned_output"
    )
    fine_tuner.train()
    fine_tuner.load_fine_tuned_model()
