# Lesson 1: Accounts & Authentication

> How user registration, login, JWT tokens, and profile management work in this project.

---

## What You'll Learn

- Django's class-based views (CBVs) for user management
- django-allauth for social/email authentication
- crispy forms for Bootstrap-styled forms
- JWT (JSON Web Tokens) with djangorestframework-simplejwt
- How the API exposes user data via DRF ViewSets

---

## 1. Django Class-Based Views (CBVs)

This project uses **CBVs** instead of function-based views. A CBV is a Python class where each HTTP method (`GET`, `POST`) maps to a method on the class.

### Example: `SignupView` (`accounts/views.py`)

```python
from django.views.generic import CreateView
from django.contrib.auth.forms import UserCreationForm

class SignupView(CreateView):
    template_name = "accounts/signup.html"
    form_class = CustomSignupForm
    success_url = reverse_lazy("accounts:login")
```

**How it works:**
1. Browser sends `GET /accounts/signup/` → Django calls `SignupView.get()` → renders `signup.html` with an empty form
2. User fills the form and submits `POST /accounts/signup/` → Django calls `SignupView.post()` → validates the form
3. If valid → creates the user → redirects to login page
4. If invalid → re-renders the form with error messages

### Key CBVs used in this project:

| View | CBV Type | Purpose |
|------|----------|---------|
| `SignupView` | `CreateView` | Create new user account |
| `LoginView` | `TemplateView` | Render login page (actual auth via allauth) |
| `LogoutView` | `RedirectView` | Log out and redirect |
| `ProfileView` | `TemplateView` | Show user profile |

---

## 2. Django Forms + Crispy Forms

Forms handle input validation and HTML rendering. This project uses **django-crispy-forms** with the Bootstrap5 pack for styled forms.

### Example: `CustomSignupForm` (`accounts/forms.py`)

```python
from django.contrib.auth.forms import UserCreationForm
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit

class CustomSignupForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.add_input(Submit('submit', 'Sign Up'))
```

**What happens:**
- `UserCreationForm` provides username + password1 + password2 fields with built-in validation
- We add `email` as a required field
- `FormHelper` + `Submit` add a styled submit button
- In the template, `{{ form|crispy }}` renders the entire form with Bootstrap styling

### Form classes in this project:

| Form | Purpose |
|------|---------|
| `CustomSignupForm` | Registration with email |
| `CustomLoginForm` | Email-based login |
| `ProfileUpdateForm` | Update username, email, bio |
| `AvatarUploadForm` | Upload profile picture |

---

## 3. Templates

Templates are HTML files that Django renders with dynamic data. Each app has its own template directory.

### Template structure:

```
backend/templates/
├── base.html                      # Base layout (navbar, footer, content block)
├── accounts/
│   ├── login.html                 # Login page
│   ├── signup.html                # Registration page
│   └── profile.html               # User profile page
├── dashboard/
│   └── dashboard.html             # Admin dashboard
```

### How template inheritance works:

**`base.html`** defines blocks:
```html
<!-- base.html -->
<html>
<body>
    {% block content %}{% endblock %}
</body>
</html>
```

**`login.html`** extends it:
```html
<!-- accounts/login.html -->
{% extends "base.html" %}
{% block content %}
  <h2>Login</h2>
  {% crispy form %}
{% endblock %}
```

### Request-Response Cycle (Template Views):

```
Browser → Daphne/WSGI → URL Router → View.get()
    → View creates Form/context → Renders template → HTML response → Browser
```

---

## 4. JWT Authentication (API Layer)

For API endpoints (used by the Next.js frontend), this project uses **JWT tokens** instead of session cookies.

### How JWT works:

```
1. POST /api/v1/auth/login/  { username, password }
   → Returns: { "access": "eyJ...", "refresh": "eyJ..." }

2. GET /api/v1/products/  Authorization: Bearer eyJ...
   → Server validates the access token
   → Returns product data

3. POST /api/v1/auth/refresh/  { "refresh": "eyJ..." }
   → Returns: { "access": "eyJ..." }  (new access token)
```

### Configuration (`config/settings/base.py`):

```python
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'AUTH_HEADER_TYPES': ('Bearer',),
}
```

- **Access token**: Short-lived (60 min). Sent with every API request in the `Authorization` header.
- **Refresh token**: Long-lived (7 days). Used only to get a new access token when the old one expires.

### User API endpoint (`api/views.py`):

```python
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]
```

This creates these URLs automatically:
- `GET /api/v1/users/` — list all users (authenticated only)
- `GET /api/v1/users/{id}/` — get a specific user
- `PUT /api/v1/users/{id}/` — update user
- `DELETE /api/v1/users/{id}/` — delete user

---

## 5. URL Routing

Each app defines its own URLs in a `urls.py` file. The root `config/urls.py` includes them all.

### Pattern:

```python
# config/urls.py
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/auth/', include('rest_framework.urls')),        # login/logout browsable API
    path('api/v1/', include('api.urls')),                         # all API endpoints
    path('accounts/', include('accounts.urls', namespace='accounts')),
    ...
]
```

### Namespacing:

```python
# accounts/urls.py
app_name = 'accounts'

urlpatterns = [
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('profile/', ProfileView.as_view(), name='profile'),
]
```

You can reference URLs in templates with: `{% url 'accounts:login' %}`

---

## 6. Complete Request-Response Walkthrough

### Scenario: User registers a new account

```
1. Browser:   GET /accounts/signup/
2. Daphne:    Routes to accounts.urls → SignupView
3. View:      get() method → creates empty CustomSignupForm → renders signup.html
4. Browser:   Displays the registration form
5. User:      Fills in username, email, passwords → clicks "Sign Up"
6. Browser:   POST /accounts/signup/  { form data }
7. View:      post() → form.is_valid() → checks passwords match, email format, etc.
8. View:      form.save() → creates User object in database
9. View:      redirect to /accounts/login/
10. Browser:  Shows login page
```

### Scenario: API login to get JWT tokens

```
1. Frontend:  POST /api/v1/auth/login/  { "username": "john", "password": "pass123" }
2. DRF:       TokenObtainPairView → validates credentials
3. DRF:       Generates access + refresh JWT tokens
4. Response:  { "access": "eyJhbGci...", "refresh": "eyJhbGci..." }
5. Frontend:  Stores tokens in localStorage/cookies
6. Frontend:  Future requests include: Authorization: Bearer eyJhbGci...
```

---

## Exercises

1. **Add a "Delete Account" view** — Create a CBV that shows a confirmation page and deletes the user on POST.
2. **Add phone number to profile** — Add a `phone` field to the User model (via a Profile model), update `ProfileUpdateForm`, and update the template.
3. **Custom JWT claim** — Add the user's email to the JWT payload by subclassing `TokenObtainPairSerializer`.

---

## Key Files

| File | What It Does |
|------|-------------|
| `apps/accounts/views.py` | 5 CBVs for auth pages |
| `apps/accounts/forms.py` | 4 crispy forms |
| `apps/accounts/urls.py` | URL routing for auth pages |
| `apps/accounts/management/commands/create_demo_users.py` | Creates test users |
| `api/views.py` → `UserViewSet` | API endpoint for user CRUD |
| `api/serializers/user_serializer.py` | User data serialization |
| `config/settings/base.py` → `SIMPLE_JWT` | JWT token configuration |
