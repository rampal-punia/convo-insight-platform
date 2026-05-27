"""v1 URL router — auto-discoverable via DRF's ``DefaultRouter``.

Every ViewSet registered here is reachable under ``/api/v1/<basename>/`` with
the standard CRUD verbs *plus* every ``@action(url_path=...)`` decorated method
exposed as ``/api/v1/<basename>/<url_path>/``.

Visit ``/api/v1/`` in a browser to see the full route map.
"""
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views_accounts import UserViewSet
from .views_analysis import (
    ConversationMetricsViewSet,
    IntentPredictionViewSet,
    LLMAgentPerformanceViewSet,
    RecommendationViewSet,
    TopicDistributionViewSet,
)
from .views_conversations import (
    AITextViewSet,
    ConversationViewSet,
    GranularEmotionViewSet,
    IntentViewSet,
    MessageViewSet,
    SentimentCategoryViewSet,
    SentimentViewSet,
    TopicViewSet,
    UserTextViewSet,
)
from .views_nlp import NLPAnalysisViewSet
from .views_orders import OrderItemViewSet, OrderTrackingViewSet, OrderViewSet
from .views_products import CategoryViewSet, ProductViewSet

router = DefaultRouter()

# Catalogue
router.register(r"products", ProductViewSet, basename="product")
router.register(r"categories", CategoryViewSet, basename="category")

# Orders
router.register(r"orders", OrderViewSet, basename="order")
router.register(r"order-items", OrderItemViewSet, basename="order-item")
router.register(r"order-tracking", OrderTrackingViewSet, basename="order-tracking")

# Conversations
router.register(r"conversations", ConversationViewSet, basename="conversation")
router.register(r"messages", MessageViewSet, basename="message")
router.register(r"user-texts", UserTextViewSet, basename="user-text")
router.register(r"ai-texts", AITextViewSet, basename="ai-text")
router.register(r"intents", IntentViewSet, basename="intent")
router.register(r"topics", TopicViewSet, basename="topic")
router.register(r"sentiments", SentimentViewSet, basename="sentiment")
router.register(r"sentiment-categories", SentimentCategoryViewSet, basename="sentiment-category")
router.register(r"granular-emotions", GranularEmotionViewSet, basename="granular-emotion")

# Analysis / insights
router.register(r"agent-performance", LLMAgentPerformanceViewSet, basename="agent-performance")
router.register(
    r"conversation-metrics", ConversationMetricsViewSet, basename="conversation-metrics"
)
router.register(r"recommendations", RecommendationViewSet, basename="recommendation")
router.register(r"topic-distributions", TopicDistributionViewSet, basename="topic-distribution")
router.register(r"intent-predictions", IntentPredictionViewSet, basename="intent-prediction")

# Accounts
router.register(r"users", UserViewSet, basename="user")

# NLP playground (function-style actions on a single ViewSet)
router.register(r"nlp", NLPAnalysisViewSet, basename="nlp")


app_name = "v1"

urlpatterns = [
    path("", include(router.urls)),
]
