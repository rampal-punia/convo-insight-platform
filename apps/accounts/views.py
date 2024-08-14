from django.shortcuts import render
from allauth.account.views import LoginView
from .forms import CustomLoginForm


class CustomLoginView(LoginView):
    template_name = 'accounts/login.html'
    form_class = CustomLoginForm
