from django.views import generic
from django.shortcuts import render


class Dashboard(generic.View):
    def get(self, *args, **kwargs):
        return render(self.request, "dashboard/dashboard.html")

    def post(self, *args, **kwargs):
        pass
