from django.contrib import admin
from .models import LLMAgentPerformance, IntentPrediction


@admin.register(IntentPrediction)
class IntentPredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'intent')
