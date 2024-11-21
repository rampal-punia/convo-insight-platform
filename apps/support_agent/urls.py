from django.urls import path
from . import views

app_name = 'support_agent'

urlpatterns = [
    path('', views.SupportAgentTemplateView.as_view(),
         name='support_agent_chat_url'),
]
