"""ViewSets for the Analysis domain."""
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters, viewsets

from analysis.models import (
    ConversationMetrics,
    IntentPrediction,
    LLMAgentPerformance,
    Recommendation,
    TopicDistribution,
)

from ..pagination import StandardResultsSetPagination
from .serializers_analysis import (
    ConversationMetricsSerializer,
    IntentPredictionSerializer,
    LLMAgentPerformanceSerializer,
    RecommendationSerializer,
    TopicDistributionSerializer,
)


class LLMAgentPerformanceViewSet(viewsets.ModelViewSet):
    queryset = LLMAgentPerformance.objects.select_related("conversation").all()
    serializer_class = LLMAgentPerformanceSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["conversation", "issue_resolved"]
    ordering_fields = ["evaluated_at", "overall_score", "quality_score"]


class ConversationMetricsViewSet(viewsets.ModelViewSet):
    queryset = ConversationMetrics.objects.select_related("conversation").all()
    serializer_class = ConversationMetricsSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["conversation"]
    ordering_fields = ["evaluated_at", "overall_satisfaction_score", "sentiment_score"]


class RecommendationViewSet(viewsets.ModelViewSet):
    queryset = Recommendation.objects.select_related("conversation").all()
    serializer_class = RecommendationSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["conversation", "is_applied"]
    ordering_fields = ["created_at", "confidence_score"]


class TopicDistributionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = TopicDistribution.objects.select_related("conversation", "topic").all()
    serializer_class = TopicDistributionSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["conversation", "topic"]
    ordering_fields = ["weight"]


class IntentPredictionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = IntentPrediction.objects.select_related("message").all()
    serializer_class = IntentPredictionSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["intent"]
    ordering_fields = ["confidence_score"]
