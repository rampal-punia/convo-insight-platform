from django.urls import path
from . import views

app_name = 'playground'

urlpatterns = [
    path("", views.NLPPlayground.as_view(), name='nlp_playground_url')
]
