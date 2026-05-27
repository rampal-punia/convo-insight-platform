from django.views.generic import TemplateView
from django.urls import reverse_lazy

from django.contrib.auth.mixins import LoginRequiredMixin


class SupportAgentTemplateView(LoginRequiredMixin, TemplateView):
    template_name = 'support_agent/support_agent_chat.html'
