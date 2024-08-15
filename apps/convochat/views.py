from django.shortcuts import render, redirect, get_object_or_404
from django.views import generic
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.urls import reverse_lazy
from .models import Conversation, Message


class FinchatView(LoginRequiredMixin, generic.View):
    def get(self, request, *args, **kwargs):
        context = {
            'segment': 'finchat',
        }
        return render(request, 'finchat/pages/finchat.html', context)


class ConversationListView(LoginRequiredMixin, generic.ListView):
    """Handle the url request for Conversation List View

    Args:
        LoginRequiredMixin: Login required
        generic : List view class from django.views.generic
    """
    model = Conversation
    template_name = 'dashboard/conversation_list.html'
    context_object_name = 'conversations'

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user).order_by('-created')


class ConversationDetailView(LoginRequiredMixin, generic.DetailView):
    model = Conversation
    template_name = 'dashboard/conversation_detail.html'
    context_object_name = 'conversation'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        message_qs = self.object.messages.order_by("created")
        return context


class ConversationDeleteView(LoginRequiredMixin, generic.DeleteView):
    model = Conversation
    template_name = 'dashboard/conversation_confirm_delete.html'
    success_url = reverse_lazy('conversation_list')

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)
