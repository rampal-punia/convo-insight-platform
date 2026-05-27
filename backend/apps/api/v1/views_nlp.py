"""ViewSet for the NLP analysis playground."""
from __future__ import annotations

import logging

from drf_spectacular.utils import extend_schema
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .serializers_nlp import NLPInputSerializer, NLPOutputSerializer

logger = logging.getLogger("playground")


class NLPAnalysisViewSet(viewsets.ViewSet):
    """NLP playground endpoints.

    Each action runs the requested ``method`` (``finetuned``, ``few_shot_learning``,
    ``rag``) for the corresponding task. Only the ``finetuned`` path is wired today
    via the existing ``ModelManager``; the other two return HTTP 501 until Step 2.
    """

    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    serializer_class = NLPInputSerializer

    # The ModelManager is heavyweight (loads BERT / BERTopic weights). Resolve it
    # lazily on first use so Django startup, migrations, and ``manage.py check``
    # don't pay the cost.
    _model_manager = None
    _sentiment_mapper = None

    def _get_model_manager(self):
        if NLPAnalysisViewSet._model_manager is None:
            from playground.consumers import ModelManager  # local import on purpose
            NLPAnalysisViewSet._model_manager = ModelManager()
        return NLPAnalysisViewSet._model_manager

    def _get_sentiment_mapper(self):
        if NLPAnalysisViewSet._sentiment_mapper is None:
            from playground.consumers import EcommerceSentimentMapper
            NLPAnalysisViewSet._sentiment_mapper = EcommerceSentimentMapper()
        return NLPAnalysisViewSet._sentiment_mapper

    def _validate(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        return serializer.validated_data

    def _ok(self, task: str, method: str, result: dict):
        payload = {"task": task, "method": method, "result": result}
        output = NLPOutputSerializer(data=payload)
        output.is_valid(raise_exception=True)
        return Response(output.validated_data)

    def _not_implemented(self, method: str):
        return Response(
            {"error": f"Method '{method}' is not yet wired in Step 1. Coming in Step 2."},
            status=status.HTTP_501_NOT_IMPLEMENTED,
        )

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------
    @extend_schema(
        summary="Sentiment analysis (e-commerce mapped)",
        request=NLPInputSerializer,
        responses=NLPOutputSerializer,
    )
    @action(detail=False, methods=["post"], url_path="sentiment", url_name="sentiment")
    def analyze_sentiment(self, request):
        data = self._validate(request)
        text, method = data["text"], data["method"]
        if method != "finetuned":
            return self._not_implemented(method)
        try:
            model = self._get_model_manager().get_model("sentiment")
            prediction = model.predict(text)
            base = prediction["predictions"][0]
            confidence = prediction["confidence"][0]
            mapper = self._get_sentiment_mapper()
            granular, overall = mapper.map_sentiment(base, confidence)
            result = {
                "label": granular,
                "score": float(confidence),
                "explanation": (
                    f"For text: '{text}'\nOverall: {overall}\n"
                    f"Granular: {granular}\nConfidence: {confidence:.2f}"
                ),
            }
            return self._ok("sentiment", method, result)
        except Exception:
            logger.exception("Error in sentiment analysis")
            return Response(
                {"error": "Error processing sentiment analysis"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ------------------------------------------------------------------
    # Intent
    # ------------------------------------------------------------------
    @extend_schema(
        summary="Intent recognition",
        request=NLPInputSerializer,
        responses=NLPOutputSerializer,
    )
    @action(detail=False, methods=["post"], url_path="intent", url_name="intent")
    def analyze_intent(self, request):
        data = self._validate(request)
        text, method = data["text"], data["method"]
        if method != "finetuned":
            return self._not_implemented(method)
        try:
            model = self._get_model_manager().get_model("intent")
            prediction = model.predict(text)
            result = {
                "label": prediction["intent"],
                "score": float(prediction["intent_confidence"]),
                "category": prediction["category"],
                "explanation": (
                    f"Intent Category: {prediction['category']}\n"
                    f"Intent: {prediction['intent']}\n"
                    f"Category Confidence: {prediction['category_confidence']}\n"
                    f"Intent Confidence: {prediction['intent_confidence']}"
                ),
            }
            return self._ok("intent", method, result)
        except Exception:
            logger.exception("Error in intent analysis")
            return Response(
                {"error": "Error processing intent analysis"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ------------------------------------------------------------------
    # Topic
    # ------------------------------------------------------------------
    @extend_schema(
        summary="Topic classification (BERTopic)",
        request=NLPInputSerializer,
        responses=NLPOutputSerializer,
    )
    @action(detail=False, methods=["post"], url_path="topic", url_name="topic")
    def analyze_topic(self, request):
        data = self._validate(request)
        text, method = data["text"], data["method"]
        if method != "finetuned":
            return self._not_implemented(method)
        try:
            bertopic_model = self._get_model_manager().get_model("topic")["bertopic"]
            text_list = [text]
            topics, probs = bertopic_model.transform(text_list)
            topic_id = topics[0]
            probability = probs[0]
            score = float(probability) if isinstance(probability, (int, float)) else float(
                probability.max()
            )
            if topic_id != -1:
                topic_words = [w for w, _ in bertopic_model.get_topic(topic_id)][:10]
                topic_repr = bertopic_model.get_representative_docs(topic_id)
            else:
                topic_words = ["No specific topic identified"]
                topic_repr = []
            result = {
                "label": topic_words,
                "score": score,
                "explanation": (
                    f"Topic ID: {topic_id}\nConfidence: {score:.2f}\n"
                    "Representative documents:\n"
                    + "\n".join(f"- {doc}..." for doc in topic_repr)
                ),
            }
            return self._ok("topic", method, result)
        except Exception:
            logger.exception("Error in topic analysis")
            return Response(
                {"error": "Error processing topic analysis"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ------------------------------------------------------------------
    # NER
    # ------------------------------------------------------------------
    @extend_schema(
        summary="Named entity recognition",
        request=NLPInputSerializer,
        responses=NLPOutputSerializer,
    )
    @action(detail=False, methods=["post"], url_path="ner", url_name="ner")
    def analyze_ner(self, request):
        data = self._validate(request)
        text, method = data["text"], data["method"]
        if method != "finetuned":
            return self._not_implemented(method)
        try:
            ner_pipeline = self._get_model_manager().get_model("ner")
            entities = ner_pipeline(text) if callable(ner_pipeline) else []
            # Normalise to plain dicts
            normalised = []
            for ent in entities:
                normalised.append(
                    {
                        "entity": ent.get("entity_group") or ent.get("entity"),
                        "word": ent.get("word"),
                        "score": float(ent.get("score", 0.0)),
                        "start": ent.get("start"),
                        "end": ent.get("end"),
                    }
                )
            result = {
                "label": normalised,
                "score": float(sum(e["score"] for e in normalised) / max(len(normalised), 1)),
                "explanation": f"Found {len(normalised)} entities in input.",
            }
            return self._ok("ner", method, result)
        except Exception:
            logger.exception("Error in NER analysis")
            return Response(
                {"error": "Error processing NER analysis"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
