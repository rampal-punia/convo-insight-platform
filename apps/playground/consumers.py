import json
import torch
from channels.generic.websocket import AsyncWebsocketConsumer
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class NLPPlaygroundConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """Accept WebSocket connection"""
        await self.accept()

        # Initialize models and tokenizers
        self.sentiment_tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.sentiment_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )

        self.intent_tokenizer = DistilBertTokenizer.from_pretrained(
            "Falconsai/intent_classification"
        )
        self.intent_model = DistilBertForSequenceClassification.from_pretrained(
            "Falconsai/intent_classification"
        )

        self.topic_tokenizer = AutoTokenizer.from_pretrained(
            "dstefa/roberta-base_topic_classification_nyt_news"
        )
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(
            "dstefa/roberta-base_topic_classification_nyt_news"
        )

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        pass

    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            task = data.get('task')
            text = data.get('text')

            if not text:
                await self.send(json.dumps({
                    'error': 'No text provided'
                }))
                return

            # Process based on selected task
            if task == 'sentiment':
                result = await self.analyze_sentiment(text)
            elif task == 'intent':
                result = await self.analyze_intent(text)
            elif task == 'topic':
                result = await self.analyze_topic(text)
            else:
                await self.send(json.dumps({
                    'error': 'Invalid task specified'
                }))
                return

            await self.send(json.dumps({
                'result': result
            }))

        except Exception as e:
            await self.send(json.dumps({
                'error': f'An error occurred: {str(e)}'
            }))

    async def analyze_sentiment(self, text):
        """Analyze sentiment of input text"""
        inputs = self.sentiment_tokenizer(
            text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)

        predicted_class_id = outputs.logits.argmax().item()
        label = self.sentiment_model.config.id2label[predicted_class_id]
        score = torch.softmax(outputs.logits, dim=1)[
            0][predicted_class_id].item()

        return {
            'label': label,
            'score': score
        }

    async def analyze_intent(self, text):
        """Analyze intent of input text"""
        inputs = self.intent_tokenizer(
            text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.intent_model(**inputs)

        predicted_class_id = outputs.logits.argmax().item()
        label = self.intent_model.config.id2label[predicted_class_id]
        score = torch.softmax(outputs.logits, dim=1)[
            0][predicted_class_id].item()

        return {
            'label': label,
            'score': score
        }

    async def analyze_topic(self, text):
        """Analyze topic of input text"""
        inputs = self.topic_tokenizer(
            text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.topic_model(**inputs)

        predicted_class_id = outputs.logits.argmax().item()
        label = self.topic_model.config.id2label[predicted_class_id]
        score = torch.softmax(outputs.logits, dim=1)[
            0][predicted_class_id].item()

        return {
            'label': label,
            'score': score
        }
