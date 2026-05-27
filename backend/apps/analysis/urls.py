# apps/analysis/urls.py

from django.urls import path
from . import views

app_name = 'analysis'

urlpatterns = [
    path('conversation/<uuid:pk>/performance/',
         views.ConversationPerformanceView.as_view(), name='conversation_performance_url'),
]
