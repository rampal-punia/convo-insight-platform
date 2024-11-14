import json
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


class ImprovedIntentClassifier(nn.Module):
    def __init__(self, base_model_name, num_categories, num_intents):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size

        # Separate embedding layers for categories and intents
        self.category_embeddings = nn.Embedding(num_categories, 64)
        self.intent_embeddings = nn.Embedding(num_intents, 64)

        # Projection layers
        self.category_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.intent_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state[:, 0, :]
        hidden_states = self.dropout(hidden_states)

        # Project BERT embeddings to the same space as label embeddings
        category_projected = self.category_projection(hidden_states)
        intent_projected = self.intent_projection(hidden_states)

        # Calculate similarity scores instead of direct classification
        category_embeddings = self.category_embeddings.weight
        intent_embeddings = self.intent_embeddings.weight

        # Compute cosine similarity
        category_logits = F.cosine_similarity(
            category_projected.unsqueeze(1),
            category_embeddings.unsqueeze(0),
            dim=2
        )

        intent_logits = F.cosine_similarity(
            intent_projected.unsqueeze(1),
            intent_embeddings.unsqueeze(0),
            dim=2
        )

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            category_loss = loss_fct(category_logits, labels[:, 0])
            intent_loss = loss_fct(intent_logits, labels[:, 1])
            loss = category_loss + intent_loss

        return {'loss': loss, 'category_logits': category_logits, 'intent_logits': intent_logits}


class IntentModelTester:
    def __init__(self, model_dir: str, base_model_name: str = "bert-base-uncased"):
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        with open(self.model_dir / "mappings.json", 'r') as f:
            self.mappings = json.load(f)

        self.model = self.load_model(base_model_name)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, base_model_name):
        model = ImprovedIntentClassifier(
            base_model_name=base_model_name,
            num_categories=len(self.mappings['category_mapping']),
            num_intents=len(self.mappings['intent_mapping'])
        )

        state_dict = load_file(self.model_dir / "model.safetensors")
        model.load_state_dict(state_dict)

        return model

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        category_probs = torch.softmax(outputs['category_logits'], dim=-1)
        intent_probs = torch.softmax(outputs['intent_logits'], dim=-1)

        category_idx = category_probs.argmax().item()
        intent_idx = intent_probs.argmax().item()

        reverse_category_mapping = {
            v: k for k, v in self.mappings['category_mapping'].items()}
        reverse_intent_mapping = {v: k for k,
                                  v in self.mappings['intent_mapping'].items()}

        return {
            'category': reverse_category_mapping[category_idx],
            'intent': reverse_intent_mapping[intent_idx],
            'category_confidence': category_probs[0][category_idx].item(),
            'intent_confidence': intent_probs[0][intent_idx].item()
        }


if __name__ == "__main__":
    model_tester = IntentModelTester("bertmodel_intent_12nov24")

    test_texts = [
        "I need to change my shipping address",
        "When will my package arrive?",
        "I want to cancel my order"
    ]

    for text in test_texts:
        prediction = model_tester.predict(text)
        print("\nInput text:", text)
        print(
            f"Predicted category: {prediction['category']} (Confidence: {prediction['category_confidence']:.2f})")
        print(
            f"Predicted intent: {prediction['intent']} (Confidence: {prediction['intent_confidence']:.2f})")
