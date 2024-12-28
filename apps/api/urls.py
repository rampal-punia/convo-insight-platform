from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import NLPAnalysisViewSet

router = DefaultRouter()
router.register(r'nlp', NLPAnalysisViewSet, basename='nlp')

urlpatterns = [
    path('', include(router.urls)),
]


'''
POST /api/nlp/sentiment/: Sentiment analysis
POST /api/nlp/intent/: Intent recognition
POST /api/nlp/topic/: Topic classification
'''
