import uuid

from django.shortcuts import render, redirect
from django.views import generic

from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse, reverse_lazy
from .models import GeneralConversation, GeneralMessage, AudioMessage


class GeneralConversationView(LoginRequiredMixin, generic.View):
    def get(self, request, *args, **kwargs):
        return render(request, 'general_assistant/generalchat.html')


class GeneralConversationListView(LoginRequiredMixin, generic.ListView):
    model = GeneralConversation
    template_name = 'general_assistant/conversation_list.html'
    context_object_name = 'conversations'

    def get_queryset(self):
        return GeneralConversation.objects.filter(user=self.request.user)


class GeneralConversationDetailView(LoginRequiredMixin, generic.DetailView):
    model = GeneralConversation
    template_name = 'general_assistant/generalchat.html'
    context_object_name = 'conversation'

    def get_object(self, queryset=None):
        conversation_id = self.kwargs.get('pk')
        if conversation_id:
            return super().get_object(queryset)
        else:
            # Create a new conversation
            return GeneralConversation.objects.create(
                user=self.request.user,
                id=uuid.uuid4(),
                status='AC'
            )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        conversation = self.get_object()
        message_qs = GeneralMessage.objects.filter(
            conversation=conversation).select_related('audio_content')

        context["previous_messages"] = message_qs
        context["conversation_id"] = conversation.id
        return context

    def get(self, request, *args, **kwargs):
        if 'pk' not in kwargs:
            # If no pk is provided, create a new conversation and redirect
            new_conversation = self.get_object()
            return redirect(reverse('general_assistant:generalchat_detail_url', kwargs={'pk': new_conversation.id}))
        return super().get(request, *args, **kwargs)


class GeneralConversationDeleteView(LoginRequiredMixin, generic.DeleteView):
    model = GeneralConversation
    template_name = 'general_assistant/conversation_confirm_delete.html'
    success_url = reverse_lazy('general_assistant:audio_list_url')
