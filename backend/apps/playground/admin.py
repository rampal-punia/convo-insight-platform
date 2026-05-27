from django.contrib import admin
from .models import RAGTextClassificationDocument


@admin.register(RAGTextClassificationDocument)
class RAGTextClassificationDocumentAdmin(admin.ModelAdmin):
    list_display = ('id', 'task', 'content')
