from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin


class NLPPlayground(LoginRequiredMixin, TemplateView):
    template_name = 'playground/nlp_playground.html'
