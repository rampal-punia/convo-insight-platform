# apps/convochat/views.py

import uuid

from django.shortcuts import render, redirect
from django.views import generic

from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse, reverse_lazy
from .models import Conversation, UserMessage, Message
from analysis.models import IntentPrediction, TopicDistribution


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
    template_name = 'convochat/convochat.html'
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
                status='AC'
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
            return redirect(reverse('convochat:conversation_detail_url', kwargs={'pk': new_conversation.id}))
        return super().get(request, *args, **kwargs)


class ConversationDeleteView(LoginRequiredMixin, generic.DeleteView):
    model = Conversation
    template_name = 'convochat/conversation_confirm_delete.html'
    success_url = reverse_lazy('convochat:conversation_list_url')


class ConversationSentimentView(LoginRequiredMixin, generic.DetailView):
    model = Conversation
    template_name = 'convochat/conversation_sentiment.html'
    context_object_name = 'conversation'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        conversation = self.get_object()
        metrices = conversation.metrics

        context['sentiment_score'] = metrices.sentiment_score if metrices else None
        return context


class ConversationIntentView(LoginRequiredMixin, generic.DetailView):
    model = Conversation
    template_name = 'convochat/conversation_intent.html'
    context_object_name = 'conversation'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        conversation = self.get_object()
        user_messages = UserMessage.objects.filter(conversation=conversation)
        intent_predictions = IntentPrediction.objects.filter(
            message__in=user_messages)

        context['intent_predictions'] = intent_predictions
        return context


class ConversationTopicView(LoginRequiredMixin, generic.DetailView):
    model = Conversation
    template_name = 'convochat/conversation_topics.html'
    context_object_name = 'conversation'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        conversation = self.get_object()
        topic_distributions = TopicDistribution.objects.filter(
            conversation=conversation).order_by('-weight')

        context['topic_distributions'] = topic_distributions
        return context
