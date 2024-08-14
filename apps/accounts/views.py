from .forms import CustomSignupForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.edit import UpdateView
from allauth.account.views import LoginView, PasswordResetView, SignupView
from . import forms


class CustomLoginView(LoginView):
    template_name = 'accounts/login.html'
    form_class = forms.CustomLoginForm


class CustomSignupView(SignupView):
    form_class = CustomSignupForm


class CustomPasswordResetView(PasswordResetView):
    form_class = forms.CustomPasswordResetForm


class CustomProfileView(LoginRequiredMixin, UpdateView):
    form_class = forms.CustomProfileForm
    template_name = 'account/profile.html'
    success_url = '/accounts/profile/'

    def get_object(self):
        return self.request.user
