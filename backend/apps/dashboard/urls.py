# apps/dashboard/urls.py

from django.urls import path
from . import views
from .health import health_check, readiness_check

app_name = 'dashboard'

urlpatterns = [
    path("", views.Dashboard.as_view(), name='home_url'),
]
