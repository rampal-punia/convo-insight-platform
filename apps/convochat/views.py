import uuid

from django.shortcuts import render, redirect
from django.views import generic
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse, reverse_lazy
from .models import Conversation, Message


class ConvoChatView(LoginRequiredMixin, generic.View):
    def get(self, request, *args, **kwargs):
        return render(request, 'convochat/convochat.html')


class ConversationListView(LoginRequiredMixin, generic.ListView):
    """Handle the url request for Conversation List View

    Args:
        LoginRequiredMixin: Login required
        generic : List view class from django.views.generic
    """
    model = Conversation
    template_name = 'convochat/conversation_list.html'
    context_object_name = 'conversations'

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)


class ConversationDetailView(LoginRequiredMixin, generic.DetailView):
    model = Conversation
    template_name = 'convochat/conversation_detail.html'
    context_object_name = 'conversation'

    def get_object(self, queryset=None):
        conversation_id = self.kwargs.get('pk')
        if conversation_id:
            return super().get_object(queryset)
        else:
            # Create a new conversation
            return Conversation.objects.create(
                user=self.request.user,
                id=uuid.uuid4(),
                status='Active'
            )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        conversation = self.get_object()
        message_qs = Message.objects.filter(
            conversation=conversation).order_by("created")

        context["previous_messages"] = message_qs
        context["conversation_id"] = conversation.id
        return context

    def get(self, request, *args, **kwargs):
        if 'pk' not in kwargs:
            # If no pk is provided, create a new conversation and redirect
            new_conversation = self.get_object()
            return redirect(reverse('conversation_detail', kwargs={'pk': new_conversation.id}))
        return super().get(request, *args, **kwargs)


class ConversationDeleteView(LoginRequiredMixin, generic.DeleteView):
    model = Conversation
    template_name = 'convochat/conversation_confirm_delete.html'
    success_url = reverse_lazy('convochat:conversation_list_url')
