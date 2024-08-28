from django.contrib import admin
from .models import Conversation, Message, UserText, AIText


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
