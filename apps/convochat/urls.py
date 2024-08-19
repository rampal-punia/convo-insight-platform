# apps/convochat/urls.py

from django.urls import path
from . import views

app_name = 'convochat'

urlpatterns = [
    path("", views.ConversationListView.as_view(), name='conversation_list_url'),
    path('chat/', views.ConversationDetailView.as_view(),
         name='new_conversation_url'),
    path('chat/<uuid:pk>/', views.ConversationDetailView.as_view(),
         name='conversation_detail_url'),
    path("<uuid:pk>/delete/", views.ConversationDeleteView.as_view(),
         name='conversation_delete_url'),
    path('chat/<uuid:pk>/sentiment/', views.ConversationSentimentView.as_view(),
         name='conversation_sentiment_url'),
    path('chat/<uuid:pk>/intent/', views.ConversationIntentView.as_view(),
         name='conversation_intent_url'),
    path('chat/<uuid:pk>/topics/', views.ConversationTopicView.as_view(),
         name='conversation_topics_url'),
    path("new/", views.ConvoChatView.as_view(), name='new_chat_url'),
]
