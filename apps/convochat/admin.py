# apps/convochat/admin.py

from django.contrib import admin
from .models import Conversation, Message, UserMessage, AIMessage


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    """
    Admin site configuration for Conversation model.
    """
    list_display = ('id', 'title', 'user',
                    'status', 'created')
    list_filter = ('created', 'modified', )
    search_fields = ('user__username', 'title',)


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    """
    Admin site configuration for Message model.
    """
    list_display = ('id', 'conversation',
                    'is_from_user', 'created')
    list_filter = (
        'is_from_user', 'conversation__user__username', 'created')
    ordering = ('-created',)


@admin.register(UserMessage)
class UserMessageAdmin(admin.ModelAdmin):
    """
    Admin site configuration for Message model.
    """
    list_display = ('id', 'conversation',
                    'is_from_user', 'created', 'sentiment_score')
    list_filter = (
        'is_from_user', 'conversation__user__username', 'created')
    ordering = ('-created',)


@admin.register(AIMessage)
class AIMessageAdmin(admin.ModelAdmin):
    """
    Admin site configuration for Message model.
    """
    list_display = ('id', 'conversation',
                    'is_from_user', 'created', 'recommendation')
    list_filter = (
        'is_from_user', 'conversation__user__username', 'created')
    ordering = ('-created',)