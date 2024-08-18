# apps/dashboard/views.py

from django.views import generic
from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin


class Dashboard(LoginRequiredMixin, generic.View):
    def get(self, *args, **kwargs):
        return render(self.request, "dashboard/dashboard.html")

    def post(self, *args, **kwargs):
        pass
