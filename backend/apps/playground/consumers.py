import os
import json
import re
import time
import logging
import traceback
import threading
from typing import List, Dict
from django.core.cache import cache
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.conf import settings
from convochat.models import (
    Intent,
    Sentiment,
    GranularEmotion,
    Topic,
    SentimentCategory,
)

logger = logging.getLogger("playground")


class ModelManager:
    _instance = None
    _lock = threading.Lock()
    _models_initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    # Initialize basic attributes
                    self.models = {}
                    self._model_locks = {
                        "sentiment": threading.Lock(),
                        "intent": threading.Lock(),
                        "topic": threading.Lock(),
                        "ner": threading.Lock(),
                    }

                    # Start model loading in background
                    self._initialize_models()
                    self._initialized = True

    def _initialize_models(self):
        """Initialize all models in a thread-safe manner.

        Each model is loaded independently so a failure in one does not
        prevent the others from loading.
        """

        def load_models():
            with self._lock:
                if not self._models_initialized:
                    for loader in (
                        self._load_sentiment_model,
                        self._load_intent_model,
                        self._load_topic_model,
                        self._load_ner_model,
                    ):
                        try:
                            loader()
                        except Exception as e:
                            logger.error(f"{loader.__name__} failed: {e}")
                            logger.error(traceback.format_exc())

                    self._models_initialized = True
                    cache.set("models_loaded", True, timeout=None)

        # Start loading in background
        thread = threading.Thread(target=load_models)
        thread.daemon = True
        thread.start()

    def _load_sentiment_model(self):
        """Load sentiment model with proper error handling"""
        from .sentiment_model_analysis import SentimentModelManager

        try:
            with self._model_locks["sentiment"]:
                if "sentiment" not in self.models:
                    logger.info("Loading sentiment model...")
                    self.models["sentiment"] = SentimentModelManager(
                        model_dir=settings.FINETUNED_MODELS["sentiment"]["path"]
                    )
                    self.models["sentiment"].load_model()
                    logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            raise

    def _load_intent_model(self):
        """Load intent model with proper error handling"""
        from .intent_recognitionwith_tr_bert import IntentModelTester

        try:
            with self._model_locks["intent"]:
                if "intent" not in self.models:
                    logger.info("Loading intent model...")
                    self.models["intent"] = IntentModelTester(
                        settings.FINETUNED_MODELS["intent"]["path"]
                    )
                    logger.info("Intent model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading intent model: {e}")
            raise

    def _load_topic_model(self):
        """Load topic model with proper error handling.

        Applies two compatibility shims for models serialised with older
        library versions:

        1. BertSdpaSelfAttention class rename (transformers < 5.x)
           The class was merged into BertSelfAttention; alias it back so
           pickled references still resolve.

        2. code() constructor signature change (Python ≤ 3.10 → Python ≥ 3.11)
           Python 3.11 added qualname (str) at position 12 and exceptiontable
           (bytes) at position 15, growing the constructor from 16 to 18 args.
           BERTopic models trained on Python ≤ 3.10 use numba whose JIT
           functions are pickled as code objects with the old 16-arg layout;
           they fail on Python 3.12 with "code() argument 13 must be str, not int".

           Fix: patch numba.core.serialize.loads (the function numba calls to
           deserialise its own pickled bytecode) with a custom Unpickler that
           intercepts builtins.code and inserts the missing arguments.
        """
        import io
        import pickle as _pickle
        import types

        # -- Shim 1: BertSdpaSelfAttention ---------------------------------
        import transformers.models.bert.modeling_bert as _bert_module

        if not hasattr(_bert_module, "BertSdpaSelfAttention"):
            _bert_module.BertSdpaSelfAttention = _bert_module.BertSelfAttention

        # -- Shim 2: code() 16-arg → 18-arg for Python ≤ 3.10 pickles -----
        def _compat_code(*args):
            """Convert 16-arg (Python ≤ 3.10) code objects to 18-arg (Python ≥ 3.11)."""
            if len(args) == 16 and isinstance(args[12], int):
                # Old layout: [..., name(11), firstlineno(12), lnotab(13), freevars(14), cellvars(15)]
                # New layout: [..., name(11), qualname(12), firstlineno(13), lnotab(14),
                #              exceptiontable(15), freevars(16), cellvars(17)]
                args = args[:12] + ("",) + args[12:14] + (b"",) + args[14:]
            return types.CodeType(*args)

        class _CompatUnpickler(_pickle.Unpickler):
            def find_class(self, module, name):
                if module == "builtins" and name == "code":
                    return _compat_code
                return super().find_class(module, name)

        def _compat_loads(data):
            return _CompatUnpickler(io.BytesIO(data)).load()

        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        try:
            with self._model_locks["topic"]:
                if "topic" not in self.models:
                    logger.info("Loading topic model...")
                    bertopic_path = settings.FINETUNED_MODELS["topic"]["bertopic_path"]
                    sentence_transformer_path = settings.FINETUNED_MODELS["topic"][
                        "transformer_path"
                    ]

                    if os.path.exists(bertopic_path) and os.path.exists(
                        sentence_transformer_path
                    ):
                        # Patch numba's internal loads for the duration of BERTopic.load()
                        # so that 16-arg code objects from Python ≤ 3.10 are converted.
                        _numba_ser = None
                        try:
                            import numba.core.serialize as _numba_ser

                            _orig_numba_loads = _numba_ser.loads
                            _numba_ser.loads = _compat_loads
                        except ImportError:
                            pass

                        try:
                            bertopic = BERTopic.load(bertopic_path)
                        finally:
                            if _numba_ser is not None:
                                _numba_ser.loads = _orig_numba_loads

                        self.models["topic"] = {
                            "bertopic": bertopic,
                            "sentence_transformer": SentenceTransformer(
                                sentence_transformer_path
                            ),
                        }
                        logger.info("Topic model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading topic model: {e}")
            raise

    def _load_ner_model(self):
        """Load NER model with proper error handling"""
        import torch
        from transformers import pipeline

        try:
            with self._model_locks["ner"]:
                if "ner" not in self.models:
                    logger.info("Loading NER model...")
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    self.models["ner"] = pipeline(
                        "ner",
                        model=settings.FINETUNED_MODELS["ner"]["path"],
                        device=device,
                        batch_size=settings.FINETUNED_MODELS.get("ner", {}).get(
                            "batch_size", 32
                        ),
                    )
                    logger.info("NER model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading NER model: {e}")
            raise

    def get_model(self, model_type):
        """Get a model instance with proper error handling and timeout"""
        try:
            if model_type not in self._model_locks:
                raise KeyError(f"Unknown model type: {model_type}")

            timeout = getattr(settings, "MODEL_LOAD_TIMEOUT", 60)
            start_time = time.time()

            while time.time() - start_time < timeout:
                with self._model_locks[model_type]:
                    if model_type in self.models:
                        return self.models[model_type]

                # If model is not loaded yet, wait a bit
                time.sleep(0.1)

            raise TimeoutError(f"Model {model_type} not loaded within timeout period")

        except Exception as e:
            logger.error(f"Error getting model {model_type}: {e}")
            logger.error(traceback.format_exc())
            raise


class EcommerceSentimentMapper:
    def __init__(self):
        # Confidence threshold
        self.threshold = 0.45

        # Mapping dictionaries
        self.granular_mapping = {
            "joy": ("Satisfaction", "Positive"),
            "love": ("Gratitude", "Positive"),
            "surprise": ("Appreciation", "Positive"),
            "sadness": ("Disappointment", "Negative"),
            "anger": ("Frustration", "Negative"),
            "fear": ("Urgency", "Negative"),
        }

    def map_sentiment(self, base_sentiment, confidence_score):
        """Map the base sentiment to e-commerce specific sentiment"""
        # Default to neutral if below threshold
        if confidence_score < self.threshold:
            return "neutral", "neutral"

        # Get the mapped sentiments (granular and general)
        return self.granular_mapping.get(base_sentiment, ("neutral", "neutral"))


class NLPPlaygroundConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """Accept WebSocket connection and initialize models"""
        from langchain_community.cache import RedisCache
        import redis as _redis
        from langchain_core.globals import set_llm_cache
        from langchain_openai import ChatOpenAI

        await self.accept()

        try:
            # Setup Redis cache for LLM
            redis_client = _redis.Redis.from_url(
                url=settings.CACHES["default"]["LOCATION"],
                db=settings.CACHES["default"]["OPTIONS"].get("db", 1),
                decode_responses=True,
            )
            set_llm_cache(RedisCache(redis_client))

            # Initialize model manager
            if not hasattr(self.__class__, "model_manager"):
                self.__class__.model_manager = ModelManager()

            self.model_manager = self.__class__.model_manager

            # Initialize ChatGPT with proper cache settings
            self.llm = ChatOpenAI(model=settings.GPT_MINI, temperature=0.1)

            # Initialize other components
            await self.initialize_components()

            # Send ready message to client
            await self.send(
                json.dumps(
                    {
                        "status": "connected",
                        "message": "WebSocket connected and initialized",
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error in connection: {e}")
            logger.error(traceback.format_exc())
            await self.send(json.dumps({"error": "Failed to initialize connection"}))
            await self.close()

    async def initialize_components(self):
        """Initialize non-model components"""
        from .text_classification_vector_store import PGVectorStoreTextClassification
        from .text_classification_rag_processor import RAGProcessorTextClassification

        try:
            # Load examples from database
            self.topic_examples = await self.load_topic_examples()
            self.intent_examples = await self.load_intent_examples()
            self.sentiment_examples = await self.load_sentiment_examples()

            # Cache sentiment choices
            self.sentiment_categories = await self.get_sentiment_categories()
            self.granular_emotions = await self.get_granular_emotions()

            # Initialize vector store and RAG processor
            self.vector_store = PGVectorStoreTextClassification()
            self.rag_processor = RAGProcessorTextClassification(
                self.vector_store, self.llm
            )

            # Setup prompts
            self.setup_prompts()

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            logger.error(traceback.format_exc())
            raise

    @database_sync_to_async
    def get_sentiment_categories(self):
        return list(SentimentCategory.objects.all().values_list("name", flat=True))

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
        from langchain_core.prompts import (
            ChatPromptTemplate,
            FewShotChatMessagePromptTemplate,
        )

        # Topic Classification Prompt
        topic_example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("assistant", "{output}")]
        )

        topic_few_shot_example_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=topic_example_prompt,
            examples=self.topic_examples,
        )

        self.topic_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a specialized topic classifier for an e-commerce customer support system. Topics are organized into categories: Product-Related, Order Management, Payment & Account, and Customer Experience. Analyze the following examples and then classify new text:",
                ),
                topic_few_shot_example_prompt,
                (
                    "human",
                    "Given the above examples, classify this text into the most appropriate topic and explain why: {input}",
                ),
            ]
        )

        # Intent Classification Prompt
        intent_example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("assistant", "{output}")]
        )

        intent_few_shot_example_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=intent_example_prompt,
            examples=self.intent_examples,
        )

        self.intent_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an intent recognition specialist for customer support. Your goal is to understand what the customer wants to achieve. Learn from these examples and then classify new text: ",
                ),
                intent_few_shot_example_prompt,
                (
                    "human",
                    "Based on these examples, classify the intent of this text and explain your reasoning: {input}",
                ),
            ]
        )

        # Sentiment Analysis Prompt with Granular Emotions
        sentiment_example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("assistant", "{output}")]
        )

        sentiment_few_shot_example_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=sentiment_example_prompt,
            examples=self.sentiment_examples,
        )

        self.sentiment_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a sentiment analysis expert for customer support interactions. You analyze both overall sentiment and specific emotional undertones. Learn from these examples and then analyze new text:",
                ),
                sentiment_few_shot_example_prompt,
                (
                    "human",
                    "Based on these examples, analyze the sentiment of this text. Provide: 1) The main sentiment (POSITIVE/NEGATIVE/NEUTRAL), 2) A granular emotion if applicable (Frustration, Satisfaction, etc.), and 3) A confidence score (0-1). If no clear sentiment is present, explicitly state NEUTRAL: {input}",
                ),
            ]
        )

    @database_sync_to_async
    def load_topic_examples(self):
        """Load topic examples from database"""
        examples = []
        topics = Topic.objects.filter(is_active=True).order_by("-usage_count")[:5]
        for topic in topics:
            examples.append(
                {
                    "input": f"Category: {topic.get_category_display()}\nText:{topic.description}",
                    "output": f"This text belongs to the topic '{topic.name}' because it discusses {topic.description[:100]}...",
                }
            )
        return examples

    @database_sync_to_async
    def load_intent_examples(self) -> List[Dict]:
        """Load intent examples from database"""
        examples = []
        intents = Intent.objects.all()
        for intent in intents:
            examples.append(
                {
                    "input": intent.description,
                    "output": f"The intent is '{intent.name}' because this text expresses {intent.description[:100]}...",
                }
            )
        return examples

    @database_sync_to_async
    def load_sentiment_examples(self) -> List[Dict]:
        """Load sentiment examples from database with granular emotions"""
        examples = []
        sentiments = Sentiment.objects.filter(confidence__gte=0.8).select_related(
            "category", "granular_emotion", "message"
        )[:5]
        for sentiment in sentiments:
            text = sentiment.message.content
            emotion_text = {
                (
                    f"with {sentiment.granular_emotion.name}"
                    if sentiment.granular_emotion
                    else ""
                )
            }
            examples.append(
                {
                    "input": text,
                    "output": f"The sentiment is {sentiment.category.name} {emotion_text} with a confidence of {sentiment.confidence:.2f} and intensity score of {sentiment.score:.2f}.",
                }
            )
        return examples

    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            task = data.get("task")
            text = data.get("text")
            # default to few_shot_learning
            method = data.get("method", "few_shot_learning")

            if not text:
                await self.send(json.dumps({"error": "No text provided"}))
                return

            if method == "finetuned":
                # Process based on selected task
                if task == "sentiment":
                    result = await self.analyze_finetuned_sentiment(text)
                elif task == "intent":
                    result = await self.analyze_finetuned_intent(text)
                elif task == "topic":
                    result = await self.analyze_finetuned_topic(text)
                elif task == "ner":  # Add NER task
                    result = await self.analyze_finetuned_ner(text)
                else:
                    await self.send(json.dumps({"error": "Invalid task specified"}))
                    return

            elif method == "few_shot_learning":
                # Process based on selected task
                if task == "sentiment":
                    result = await self.analyze_sentiment(text)
                elif task == "intent":
                    result = await self.analyze_intent(text)
                elif task == "topic":
                    result = await self.analyze_topic(text)
                else:
                    await self.send(json.dumps({"error": "Invalid task specified"}))
                    return

            elif method == "rag":
                # Map task to rag task types
                task_map = {"sentiment": "SE", "intent": "IN", "topic": "TO"}
                rag_task = task_map[task]

                # Process using RAG
                result = await self.rag_processor.process(text, rag_task)

            await self.send(json.dumps({"result": result}))

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            await self.send(json.dumps({"error": f"An error occurred: {str(e)}"}))

    async def analyze_finetuned_sentiment(self, text):
        """Analyze sentiment with proper error handling and state management"""
        try:
            sentiment_model = self.model_manager.get_model("sentiment")
            prediction_results = sentiment_model.predict(text)

            # Extract results
            base_sentiment = prediction_results["predictions"][0]
            confidence_score = prediction_results["confidence"][0]

            # Map sentiment
            mapper = EcommerceSentimentMapper()
            granular_sentiment, sentiment = mapper.map_sentiment(
                base_sentiment, confidence_score
            )

            # Create explanation
            explanation = (
                f"For the text: '{text}'\n"
                f"Overall sentiment: {sentiment}\n"
                f"Granular Sentiment: {granular_sentiment}\n"
                f"Confidence: {confidence_score:.2f}"
            )

            return {
                "label": granular_sentiment,
                "score": str(confidence_score),
                "explanation": explanation,
            }

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            logger.error(traceback.format_exc())
            raise

    async def analyze_finetuned_intent(self, text):
        """Analyze intent with proper error handling and state management"""
        try:
            intent_model = self.model_manager.get_model("intent")
            prediction = intent_model.predict(text)

            return {
                "label": prediction["intent"],
                "score": prediction["intent_confidence"],
                "explanation": (
                    f"Intent Category: {prediction['category']}\n"
                    f"Intent Sub-Category: {prediction['intent']}\n"
                    f"Category Confidence: {prediction['category_confidence']}\n"
                    f"Intent Confidence: {prediction['intent_confidence']}"
                ),
                "category": prediction["category"],
            }
        except Exception as e:
            logger.error(f"Error in intent analysis: {e}")
            logger.error(traceback.format_exc())
            raise

    async def analyze_finetuned_topic(self, text: str):
        """Analyze topic of input text"""
        try:
            # Ensure text is a list for BERTopic and handle empty inputs
            if not text:
                return {"label": ["No text provided"], "score": 0.0}

            if isinstance(text, str):
                text = [text]

            # Get topic prediction and probability
            bertopic_model = self.model_manager.get_model("topic")["bertopic"]
            topics, probs = bertopic_model.transform(text)
            # Get first topic since we only passed one text
            topic_id = topics[0]
            probability = probs[0]  # Get first probability

            # Get topic information
            if topic_id != -1:  # -1 indicates no topic assigned
                topic_words = [word for word, _ in bertopic_model.get_topic(topic_id)][
                    :10
                ]
                topic_repr = bertopic_model.get_representative_docs(topic_id)
            else:
                topic_words = ["No specific topic identified"]
                topic_repr = []

            return {
                "label": topic_words,
                "score": (
                    float(probability)
                    if isinstance(probability, (float, int))
                    else float(probability.max())
                ),
                "explanation": (
                    f"Topic ID: {topic_id}\n"
                    f"Confidence: {float(probability) if isinstance(probability, (float, int)) else float(probability.max()):.2f}\n"
                    f"Representative documents:\n"
                    + "\n".join(f"- {doc}..." for doc in topic_repr)
                ),
            }
        except Exception as e:
            logger.error(f"Error in topic analysis: {e}")
            logger.error(traceback.format_exc())
            return {
                "label": ["Error analyzing topic"],
                "score": 0.0,
                "explanation": f"An error occurred: {str(e)}",
            }

    @database_sync_to_async
    def get_granular_emotions(self):
        return list(GranularEmotion.objects.all().values_list("name", flat=True))

    async def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using ChatGPT with few-shot learning"""
        prompt = self.sentiment_prompt.format(input=text)
        logger.info(f"final sentiment prompt is: {prompt}")
        response = await self.llm.ainvoke(prompt)
        content = response.content.strip().upper()

        # Parse the structured response
        result = {
            "label": "NEUTRAL",  # Default
            "score": 0.5,
            "explanation": response.content,
            "granular_emotion": None,
        }

        # Extract main sentiment
        if "POSITIVE" in content:
            result["label"] = "POSITIVE"
            result["score"] = 0.9
        elif "NEGATIVE" in content:
            result["label"] = "NEGATIVE"
            result["score"] = 0.9
        elif "NEUTRAL" in content:
            result["label"] = "NEUTRAL"
            result["score"] = 0.7
        else:
            # If no clear sentiment is found, keep default NEUTRAL with lower confidence
            result["score"] = 0.5

        # Extract confidence score if present
        confidence_matches = re.findall(r"CONFIDENCE[:\s]+(\d*\.?\d+)", content)
        if confidence_matches:
            try:
                result["score"] = float(confidence_matches[0])
            except ValueError:
                pass  # Keep default score if conversion fails

        # Extract granular emotion if present
        granular_emotions = await self.get_granular_emotions()
        for emotion in granular_emotions:
            if emotion.upper() in content:
                result["granular_emotion"] = emotion[0]
                break

        return result

    async def analyze_intent(self, text: str) -> Dict:
        """Analyze intent using ChatGPT with few-shot learning"""
        prompt = self.intent_prompt.format(input=text)
        logger.debug("Final intent prompt: %s", prompt)
        response = await self.llm.ainvoke(prompt)

        # Extract intent from response
        content = response.content.strip()
        result = {
            "label": content.split("'")[1] if "'" in content else content,
            "score": 0.9 if "'" in content else 0.7,
            "explanation": response.content,
        }

        # Verify against known intents
        intents = await self.get_intents()
        intent_names = [intent.name for intent in intents]
        if not any(
            intent.lower() in result["label"].lower() for intent in intent_names
        ):
            result["score"] = 0.6  # Lower confidence for novel classification

        return result

    async def analyze_topic(self, text: str) -> Dict:
        """Analyze topic using ChatGPT with few-shot learning"""
        prompt = self.topic_prompt.format(input=text)
        logger.debug("Final topic prompt: %s", prompt)
        response = await self.llm.ainvoke(prompt)

        content = response.content.strip()
        result = {
            "label": content.split("'")[1] if "'" in content else content,
            "score": 0.9 if "'" in content else 0.7,
            "explanation": response.content,
            "category": None,
        }

        # Extract category if mentioned
        for category in Topic.Category.choices:
            if category[1].upper() in content.upper():
                result["category"] = category[0]

        return result

    async def analyze_finetuned_ner(self, text):
        """Analyze named entities with proper error handling and state management"""
        try:
            ner_model = self.model_manager.get_model("ner")

            # Process text in batches if it's too long
            max_length = 512  # Maximum sequence length for BERT
            words = text.split()
            processed_entities = []

            for i in range(0, len(words), max_length):
                batch_text = " ".join(words[i : i + max_length])
                predictions = ner_model(batch_text)

                # Process and merge consecutive entity tokens
                current_entity = None

                for pred in predictions:
                    # Clean the word (remove special tokens)
                    clean_word = pred["word"].replace("##", "")
                    entity_type = pred["entity"].split("-")[-1]  # Remove B-/I- prefixes

                    if (
                        current_entity
                        and current_entity["entity"] == entity_type
                        and pred["start"] - current_entity["end"] <= 1
                    ):
                        # Merge with current entity
                        current_entity["word"] += clean_word
                        current_entity["end"] = pred["end"]
                        current_entity["score"] = (
                            current_entity["score"] + float(pred["score"])
                        ) / 2
                    else:
                        # Save current entity if exists
                        if current_entity:
                            processed_entities.append(current_entity)

                        # Start new entity
                        current_entity = {
                            "entity": entity_type,
                            "word": clean_word,
                            "score": float(pred["score"]),
                            "start": pred["start"],
                            "end": pred["end"],
                        }

                # Add last entity if exists
                if current_entity:
                    processed_entities.append(current_entity)

            # Group entities by type with descriptions
            entity_descriptions = {
                "PER": "Person",
                "ORG": "Organization",
                "LOC": "Location",
                "MISC": "Miscellaneous",
                "DATE": "Date",
                "TIME": "Time",
                "MONEY": "Money",
                "PERCENT": "Percentage",
            }

            entities_by_type = {}
            for entity in processed_entities:
                entity_type = entity["entity"]
                type_name = entity_descriptions.get(entity_type, entity_type)

                if type_name not in entities_by_type:
                    entities_by_type[type_name] = []
                entities_by_type[type_name].append(entity)

            # Create detailed explanation
            explanation = f"Found {len(processed_entities)} entities in the text:\n"
            for type_name, entities in entities_by_type.items():
                explanation += f"\n{type_name}:\n"
                for entity in entities:
                    confidence = entity["score"] * 100
                    explanation += (
                        f"- {entity['word']} (Confidence: {confidence:.1f}%)\n"
                    )
                    if type_name == "Person":
                        explanation += "  Role/Context: Mentioned in text\n"
                    elif type_name == "Organization":
                        explanation += "  Type: Company/Institution\n"

            # Format for frontend display
            return {
                "entities": processed_entities,
                "entities_by_type": entities_by_type,
                "explanation": explanation,
                "original_text": text,
                "entities_with_positions": [
                    {
                        "text": entity["word"],
                        "type": entity_descriptions.get(
                            entity["entity"], entity["entity"]
                        ),
                        "start": entity["start"],
                        "end": entity["end"],
                        "confidence": entity["score"],
                    }
                    for entity in processed_entities
                ],
            }

        except Exception as e:
            logger.error(f"Error in NER analysis: {e}")
            logger.error(traceback.format_exc())
            raise
