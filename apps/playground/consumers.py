import os
import json
from typing import List, Dict
import torch
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from django.db import transaction
from django.conf import settings
from convochat.models import Intent, Sentiment, GranularEmotion, Topic, SentimentCategory
from .text_classification_vector_store import PGVectorStoreTextClassification
from .text_classification_rag_processor import RAGProcessorTextClassification
from .intent_recognitionwith_tr_bert import IntentModelTester
from .sentiment_model_analysis import SentimentModelManager

# Force CPU if CUDA is unavailable
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NLPPlaygroundConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """Accept WebSocket connection"""
        await self.accept()

        # Initialize fine-tuned models and tokenizers with error handling
        try:
            self.sentiment_tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.sentiment_model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.granular_sentiment_model = SentimentModelManager(
                model_dir="apps/playground/sentiment_tr_model")
            self.granular_sentiment_model.load_model()
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            self.sentiment_model = None

        try:
            # self.intent_tokenizer = DistilBertTokenizer.from_pretrained(
            #     "Falconsai/intent_classification"
            # )
            # self.intent_model = DistilBertForSequenceClassification.from_pretrained(
            #     "Falconsai/intent_classification"
            # )
            self.model_tester = IntentModelTester(
                "apps/playground/bertmodel_intent_12nov24")
        except Exception as e:
            print(f"Error loading intent model: {e}")
            self.intent_model = None

        try:
            bertopic_path = "/home/ram/convo-insight-platform/apps/playground/fine_tuned_sentence_transformer/trained_bertopic_transformer_model"
            sentence_transformer_path = "/home/ram/convo-insight-platform/apps/playground/fine_tuned_sentence_transformer"

            if os.path.exists(bertopic_path) and os.path.exists(sentence_transformer_path):
                self.topic_model = BERTopic.load(bertopic_path)
                self.sentence_model = SentenceTransformer(
                    sentence_transformer_path)
            # else:
            #     print("Model paths not found, initializing with defaults")
            #     self.topic_model = None
            #     self.sentence_model = None
        except Exception as e:
            print(f"Error loading topic model: {e}")
            self.topic_model = None
            self.sentence_model = None

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.GPT_MINI,
            temperature=0.1
        )

        # Load examples from database with error handling
        try:
            self.topic_examples = await self.load_topic_examples()
            self.intent_examples = await self.load_intent_examples()
            self.sentiment_examples = await self.load_sentiment_examples()
        except Exception as e:
            print(f"Error loading examples: {e}")
            self.topic_examples = []
            self.intent_examples = []
            self.sentiment_examples = []

        # Cache sentiment choices
        try:
            self.sentiment_categories = await self.get_sentiment_categories()
            self.granular_emotions = await self.get_granular_emotions()
        except Exception as e:
            print(f"Error loading sentiment categories: {e}")
            self.sentiment_categories = []
            self.granular_emotions = []

        self.vector_store = PGVectorStoreTextClassification()
        self.rag_processor = RAGProcessorTextClassification(
            self.vector_store, self.llm)

        # Initialize prompts
        self.setup_prompts()

    @database_sync_to_async
    def get_sentiment_categories(self):
        return list(SentimentCategory.objects.all().values_list('name', flat=True))

    @database_sync_to_async
    def get_topics(self):
        """Get topics from database"""
        return list(Topic.objects.filter(is_active=True))

    @database_sync_to_async
    def get_intents(self) -> List[Intent]:
        """Get intents from database"""
        return list(Intent.objects.all())

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        pass

    def setup_prompts(self):
        """Setup few-shot prompts for each task"""

        # Topic Classification Prompt
        topic_example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("assistant", "{output}")
        ])

        topic_few_shot_example_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=topic_example_prompt,
            examples=self.topic_examples,
        )

        self.topic_prompt = ChatPromptTemplate.from_messages(
            [
                ('system',
                 "You are a specialized topic classifier for an e-commerce customer support system. Topics are organized into categories: Product-Related, Order Management, Payment & Account, and Customer Experience. Analyze the following examples and then classify new text:"),
                topic_few_shot_example_prompt,
                ('human',
                 "Given the above examples, classify this text into the most appropriate topic and explain why: {input}"),
            ]
        )

        # Intent Classification Prompt
        intent_example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("assistant", "{output}")
        ])

        intent_few_shot_example_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=intent_example_prompt,
            examples=self.intent_examples,
        )

        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intent recognition specialist for customer support. Your goal is to understand what the customer wants to achieve. Learn from these examples and then classify new text: "),
            intent_few_shot_example_prompt,
            ("human",
             "Based on these examples, classify the intent of this text and explain your reasoning: {input}")
        ])

        # Sentiment Analysis Prompt with Granular Emotions
        sentiment_example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("assistant", "{output}")
        ])

        sentiment_few_shot_example_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=sentiment_example_prompt,
            examples=self.sentiment_examples,
        )

        self.sentiment_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a sentiment analysis expert for customer support interactions. You analyze both overall sentiment and specific emotional undertones. Learn from these examples and then analyze new text:"),
            sentiment_few_shot_example_prompt,
            ("human",
             "Based on these examples, analyze the sentiment of this text. Provide: 1) The main sentiment (POSITIVE/NEGATIVE/NEUTRAL), 2) A granular emotion if applicable (Frustration, Satisfaction, etc.), and 3) A confidence score (0-1): {input}")
        ])

    @database_sync_to_async
    def load_topic_examples(self):
        """Load topic examples from database"""
        examples = []
        topics = Topic.objects.filter(
            is_active=True).order_by('-usage_count')[:5]
        for topic in topics:
            examples.append({
                "input": f"Category: {topic.get_category_display()}\nText:{topic.description}",
                "output": f"This text belongs to the topic '{topic.name}' because it discusses {topic.description[:100]}..."
            })
        return examples

    @database_sync_to_async
    def load_intent_examples(self) -> List[Dict]:
        """Load intent examples from database"""
        examples = []
        intents = Intent.objects.all()
        for intent in intents:
            examples.append({
                "input": intent.description,
                "output": f"The intent is '{intent.name}' because this text expresses {intent.description[:100]}..."
            })
        return examples

    @database_sync_to_async
    def load_sentiment_examples(self) -> List[Dict]:
        """Load sentiment examples from database with granular emotions"""
        examples = []
        sentiments = (Sentiment.objects
                      .filter(confidence__gte=0.8)
                      .select_related('category', 'granular_emotion', 'message')[:5])
        for sentiment in sentiments:
            text = sentiment.message.content
            emotion_text = {f"with {sentiment.granular_emotion.name}"
                            if sentiment.granular_emotion else ""}
            examples.append({
                "input": text,
                "output": f"The sentiment is {sentiment.category.name} {emotion_text} with a confidence of {sentiment.confidence:.2f} and intensity score of {sentiment.score:.2f}."
            })
        return examples

    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        # try:
        data = json.loads(text_data)
        task = data.get('task')
        text = data.get('text')
        # default to few_shot_learning
        method = data.get('method', 'few_shot_learning')

        if not text:
            await self.send(json.dumps({
                'error': 'No text provided'
            }))
            return

        if method == 'finetuned':
            # Process based on selected task
            if task == 'sentiment':
                result = await self.analyze_finetuned_sentiment(text)
            elif task == 'intent':
                result = await self.analyze_finetuned_intent(text)
            elif task == 'topic':
                result = await self.analyze_finetuned_topic(text)
            else:
                await self.send(json.dumps({
                    'error': 'Invalid task specified'
                }))
                return

        elif method == 'few_shot_learning':
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

        elif method == 'rag':
            # Map task to rag task types
            task_map = {
                'sentiment': 'SE',
                'intent': 'IN',
                'topic': 'TO'
            }
            rag_task = task_map[task]

            # Process using RAG
            result = await self.rag_processor.process(text, rag_task)

        await self.send(json.dumps({
            'result': result
        }))

        # except Exception as e:
        #     print(str(e))
        #     await self.send(json.dumps({
        #         'error': f'An error occurred: {str(e)}'
        #     }))

    async def analyze_finetuned_sentiment(self, text):
        """Analyze sentiment of input text"""
        inputs = self.sentiment_tokenizer(
            text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)

        predicted_class_id = outputs.logits.argmax().item()
        label = self.sentiment_model.config.id2label[predicted_class_id]
        score = torch.softmax(outputs.logits, dim=1)[
            0][predicted_class_id].item()

        granular_sentiment = self.granular_sentiment_model.predict(text)

        return {
            'label': label,
            'score': score,
            'explanation': f"Granular Sentiment: {granular_sentiment['predictions'][0]} | {granular_sentiment['confidence'][0]}"
        }

    async def analyze_finetuned_intent(self, text):
        """Analyze intent of input text"""
        prediction = self.model_tester.predict(text)
        print("*"*40)
        print(prediction)
        print("*"*40)

        return {
            'label': prediction['intent'],
            'score': prediction['category_confidence'],
            'explanation': f"Intent Category: {prediction['category']} | Intent Sub-Category: {prediction['intent']} | Intent Category Confidence: {prediction['intent_confidence']}",
        }

    # async def analyze_finetuned_intent(self, text):
    #     """Analyze intent of input text"""
    #     inputs = self.intent_tokenizer(
    #         text, return_tensors="pt", truncation=True)
    #     with torch.no_grad():
    #         outputs = self.intent_model(**inputs)

    #     predicted_class_id = outputs.logits.argmax().item()
    #     label = self.intent_model.config.id2label[predicted_class_id]
    #     score = torch.softmax(outputs.logits, dim=1)[
    #         0][predicted_class_id].item()

    #     return {
    #         'label': label,
    #         'score': score
    #     }

    async def analyze_finetuned_topic(self, text: str):
        """Analyze topic of input text"""
        # Ensure text is a list for BERTopic
        if isinstance(text, str):
            text = [text]

        # Get topic prediction and probability
        topics, probs = self.topic_model.transform(text)
        topic_id = topics[0]  # Get first topic since we only passed one text
        probability = probs[0]  # Get first probability

        # Get topic information
        if topic_id != -1:  # -1 indicates no topic assigned
            topic_words = [word for word,
                           _ in self.topic_model.get_topic(topic_id)][:10]
            topic_repr = self.topic_model.get_representative_docs(topic_id)
        else:
            topic_words = []
            topic_repr = []

        return {
            "label": topic_words,
            # Convert numpy float to Python float
            "score": float(probability)
            # "label": [topic_id, topic_words],
            # # Convert numpy float to Python float
            # "score": (float(probability), topic_repr[:3]),
        }

    async def _analyze_finetuned_topic(self, text: str):
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

    @database_sync_to_async
    def get_granular_emotions(self):
        return list(GranularEmotion.objects.all().values_list('name', flat=True))

    async def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using ChatGPT with few-shot learning"""
        prompt = self.sentiment_prompt.format(input=text)
        print("*"*40)
        print("final sentiment prompt is: ", prompt)
        response = await self.llm.ainvoke(prompt)
        content = response.content.strip().upper()

        # Parse the structured response
        result = {
            'label': 'NEUTRAL',  # Default
            'score': 0.5,
            'explanation': response.content,
            'granular_emotion': None
        }

        # Extract main sentiment
        if "POSITIVE" in content:
            result['label'] = "POSITIVE"
            result['score'] = 0.9
        elif "NEGATIVE" in content:
            result['label'] = "NEGATIVE"
            result['score'] = 0.9

        # Extract granular emotion if present
        granular_emotions = await self.get_granular_emotions()
        for emotion in granular_emotions:
            if emotion.upper() in content:
                result['granular_emotion'] = emotion[0]
                break

        return result

    async def analyze_intent(self, text: str) -> Dict:
        """Analyze intent using ChatGPT with few-shot learning"""
        prompt = self.intent_prompt.format(input=text)
        print("*"*40)
        print("final intent prompt is: ", prompt)
        response = await self.llm.ainvoke(prompt)

        # Extract intent from response
        content = response.content.strip()
        result = {
            'label': content.split("'")[1] if "'" in content else content,
            'score': 0.9 if "'" in content else 0.7,
            'explanation': response.content
        }

        # Verify against known intents
        intents = await self.get_intents()
        intent_names = [intent.name for intent in intents]
        if not any(intent.lower() in result['label'].lower() for intent in intent_names):
            result['score'] = 0.6  # Lower confidence for novel classification

        return result

    async def analyze_topic(self, text: str) -> Dict:
        """Analyze topic using ChatGPT with few-shot learning"""
        prompt = self.topic_prompt.format(input=text)
        print("*"*40)
        print("final topic prompt is: ", prompt)
        response = await self.llm.ainvoke(prompt)

        content = response.content.strip()
        result = {
            'label': content.split("'")[1] if "'" in content else content,
            'score': 0.9 if "'" in content else 0.7,
            'explanation': response.content,
            'category': None
        }

        # Extract category if mentioned
        for category in Topic.Category.choices:
            if category[1].upper() in content.upper():
                result['category'] = category[0]

        return result
