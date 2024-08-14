from allauth.account.forms import LoginForm
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Field


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
