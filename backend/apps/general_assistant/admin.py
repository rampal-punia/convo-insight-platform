from django.contrib import admin
from .models import GeneralConversation, GeneralMessage, AudioMessage


@admin.register(GeneralConversation)
class GeneralConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'user',
                    'status', 'created')
    list_display_links = ('id', 'title', 'status')
    list_filter = ('created', 'modified', )
    search_fields = ('user__username', 'title',)


@admin.register(GeneralMessage)
class GeneralMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'conversation',
                    'is_from_user', 'content_type', 'created')
    list_display_links = ('id', 'conversation', 'content_type')
    list_filter = (
        'is_from_user', 'conversation__user__username', 'created')
    ordering = ('-created',)


@admin.register(AudioMessage)
class AudioMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'message')
