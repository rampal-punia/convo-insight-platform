from django.contrib.auth import get_user_model
from django import forms
from allauth.account.forms import LoginForm, SignupForm, ResetPasswordForm
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Field

User = get_user_model()


class CustomLoginForm(LoginForm):
    def __init__(self, *args, **kwargs):
        super(CustomLoginForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_method = 'post'
        self.helper.layout = Layout(
            Field('login', placeholder="Email", css_class='mb-3'),
            Field('password', placeholder="Password", css_class='mb-3'),
            Submit('submit', 'Log In', css_class='btn btn-primary btn-block')
        )


class CustomSignupForm(SignupForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_show_labels = False
        self.helper.form_method = 'post'
        self.helper.layout = Layout(
            Field('email', placeholder="Email", css_class='mb-3'),
            Field('password1', placeholder="Password", css_class='mb-3'),
            Field('password2', placeholder="Confirm Password", css_class='mb-3'),
            Field('submit', "Sign Up", css_class='btn btn-primary btn-block'),
        )


class CustomPasswordResetForm(ResetPasswordForm):
    def __init__(self, *args, **kwargs):
        super(CustomPasswordResetForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_show_labels = False
        self.helper.layout = Layout(
            Field('email', placeholder="Email", css_class='mb-3'),
            Submit('submit', 'Reset Password',
                   css_class='btn btn-primary btn-block')
        )


class CustomProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

    def __init__(self, *args, **kwargs):
        super(CustomProfileForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        self.helper.form_show_labels = False
        self.helper.layout = Layout(
            Field('first_name', placeholder="First Name", css_class='mb-3'),
            Field('last_name', placeholder="Last Name", css_class='mb-3'),
            Field('email', placeholder="Email", css_class='mb-3'),
            Submit('submit', 'Update Profile',
                   css_class='btn btn-primary btn-block')
        )
