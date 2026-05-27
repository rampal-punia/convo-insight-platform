from django.contrib import admin
from .models import (
    Conversation,
    Message,
    UserText,
    AIText,
    Intent,
    Topic,
    SentimentCategory,
    Sentiment,
    GranularEmotion,
)


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


@admin.register(UserText)
class UserTextAdmin(admin.ModelAdmin):
    """
    Admin site configuration for Message model.
    """
    list_display = ('id', 'message')
    list_filter = ('id', 'message')


@admin.register(AIText)
class AITextAdmin(admin.ModelAdmin):
    """
    Admin site configuration for Message model.
    """
    list_display = ('id', 'message')
    list_filter = ('id', 'message')


@admin.register(Intent)
class IntentAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'description')


@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'description')


@admin.register(SentimentCategory)
class SentimentCategoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'description', 'priority_weight')


@admin.register(GranularEmotion)
class GranularEmotionAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'description', 'associated_sentiment')


@admin.register(Sentiment)
class SentimentAdmin(admin.ModelAdmin):
    list_display = ('id', 'category', 'granular_emotion', 'message')
