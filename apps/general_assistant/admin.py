# apps/audio_interface/admin.py

from django.contrib import admin
from .models import AudioMessage


@admin.register(AudioMessage)
class AudioMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'message')
