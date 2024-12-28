from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.core.cache import cache
from playground.consumers import ModelManager, EcommerceSentimentMapper
from .serializers import NLPInputSerializer, NLPOutputSerializer
import logging

logger = logging.getLogger('orders')


class NLPAnalysisViewSet(viewsets.ViewSet):
    """
    API endpoint for NLP analysis tasks
    """
    serializer_class = NLPInputSerializer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_manager = ModelManager()

    @action(detail=False, methods=['post'], url_path='sentiment')
    def analyze_sentiment(self, request):
        """
        Analyze sentiment of input text
        """
        serializer = self.serializer_class(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        text = serializer.validated_data['text']
        method = serializer.validated_data['method']

        try:
            if method == 'finetuned':
                sentiment_model = self.model_manager.get_model('sentiment')
                prediction_results = sentiment_model.predict(text)

                base_sentiment = prediction_results['predictions'][0]
                confidence_score = prediction_results['confidence'][0]

                mapper = EcommerceSentimentMapper()
                granular_sentiment, sentiment = mapper.map_sentiment(
                    base_sentiment, confidence_score
                )

                result = {
                    'label': granular_sentiment,
                    'score': str(confidence_score),
                    'explanation': (
                        f"For the text: '{text}'\n"
                        f"Overall sentiment: {sentiment}\n"
                        f"Granular Sentiment: {granular_sentiment}\n"
                        f"Confidence: {confidence_score:.2f}"
                    )
                }

            output_serializer = NLPOutputSerializer(data={'result': result})
            if output_serializer.is_valid():
                return Response(output_serializer.validated_data)
            return Response(output_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return Response(
                {'error': 'Error processing sentiment analysis'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'], url_path='intent')
    def analyze_intent(self, request):
        """
        Analyze intent of input text
        """
        serializer = self.serializer_class(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        text = serializer.validated_data['text']
        method = serializer.validated_data['method']

        try:
            if method == 'finetuned':
                intent_model = self.model_manager.get_model('intent')
                prediction = intent_model.predict(text)

                result = {
                    'label': prediction['intent'],
                    'score': prediction['intent_confidence'],
                    'explanation': (
                        f"Intent Category: {prediction['category']}\n"
                        f"Intent Sub-Category: {prediction['intent']}\n"
                        f"Category Confidence: {prediction['category_confidence']}\n"
                        f"Intent Confidence: {prediction['intent_confidence']}"
                    ),
                    'category': prediction['category']
                }

            output_serializer = NLPOutputSerializer(data={'result': result})
            if output_serializer.is_valid():
                return Response(output_serializer.validated_data)
            return Response(output_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"Error in intent analysis: {str(e)}")
            return Response(
                {'error': 'Error processing intent analysis'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'], url_path='topic')
    def analyze_topic(self, request):
        """
        Analyze topic of input text
        """
        serializer = self.serializer_class(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        text = serializer.validated_data['text']
        method = serializer.validated_data['method']

        try:
            if method == 'finetuned':
                if not text:
                    return Response(
                        {'error': 'No text provided'},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                text_list = [text] if isinstance(text, str) else text

                bertopic_model = self.model_manager.get_model('topic')[
                    'bertopic']
                topics, probs = bertopic_model.transform(text_list)

                topic_id = topics[0]
                probability = probs[0]

                if topic_id != -1:
                    topic_words = [word for word,
                                   _ in bertopic_model.get_topic(topic_id)][:10]
                    topic_repr = bertopic_model.get_representative_docs(
                        topic_id)
                else:
                    topic_words = ["No specific topic identified"]
                    topic_repr = []

                result = {
                    "label": topic_words,
                    "score": float(probability) if isinstance(probability, (float, int)) else float(probability.max()),
                    "explanation": (
                        f"Topic ID: {topic_id}\n"
                        f"Confidence: {float(probability) if isinstance(probability, (float, int)) else float(probability.max()):.2f}\n"
                        f"Representative documents:\n" +
                        "\n".join(f"- {doc}..." for doc in topic_repr)
                    )
                }

            output_serializer = NLPOutputSerializer(data={'result': result})
            if output_serializer.is_valid():
                return Response(output_serializer.validated_data)
            return Response(output_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"Error in topic analysis: {str(e)}")
            return Response(
                {'error': 'Error processing topic analysis'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
