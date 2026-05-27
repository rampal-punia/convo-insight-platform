# apps/accounts/views.py

from .forms import CustomSignupForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.edit import UpdateView
from django.urls import reverse_lazy
from allauth.account.views import LoginView, PasswordResetView, SignupView
from django.contrib.auth.views import LogoutView
from . import forms


class CustomLoginView(LoginView):
    template_name = 'accounts/login.html'
    form_class = forms.CustomLoginForm


class CustomSignupView(SignupView):
    template_name = 'accounts/signup.html'
    form_class = CustomSignupForm
    success_url = '/'


class CustomPasswordResetView(PasswordResetView):
    template_name = 'accounts/confirm_passwordreset.html'
    form_class = forms.CustomPasswordResetForm


class CustomProfileView(LoginRequiredMixin, UpdateView):
    form_class = forms.CustomProfileForm
    template_name = 'accounts/profile.html'
    success_url = '/accounts/profile/'

    def get_object(self):
        return self.request.user

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["account_logout"] = reverse_lazy('account_logout')
        return context


class CustomLogoutView(LogoutView):
    next_page = reverse_lazy('account_login')
