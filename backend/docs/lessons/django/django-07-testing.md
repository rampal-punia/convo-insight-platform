# Quick Win 07: Testing — Write Tests, Sleep Better

> Tests catch bugs before they reach production. This project uses pytest + pytest-django.

---

## Test Setup

```python
# pytest.ini (or setup.cfg)
[pytest]
DJANGO_SETTINGS_MODULE = config.settings.base
asyncio_mode = auto
```

Tests live in each app's `tests/` directory:

```
apps/general_assistant/
├── tests/
│   ├── __init__.py
│   ├── test_models.py       # Model tests
│   └── test_services.py     # Service logic tests
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run one file
pytest apps/general_assistant/tests/test_models.py

# Run one test class
pytest apps/general_assistant/tests/test_models.py::TestGeneralConversation

# Run one test method
pytest apps/general_assistant/tests/test_services.py::TestQueryImageModel::test_success

# Run with verbose output
pytest -v

# Run with print statements visible
pytest -s

# Run only failed tests from last run
pytest --lf
```

---

## Test Types

### 1. Model Tests — Test database models

```python
# apps/general_assistant/tests/test_models.py
import pytest
from general_assistant.models import GeneralConversation, GeneralMessage

pytestmark = pytest.mark.django_db  # Need database access

class TestGeneralConversation:
    def test_create_conversation(self, user):
        conv = GeneralConversation.objects.create(user=user, title="Test Chat")
        assert conv.title == "Test Chat"
        assert conv.user == user
        assert conv.created_at is not None

    def test_default_title(self, user):
        conv = GeneralConversation.objects.create(user=user)
        assert conv.title == "Untitled Conversation"

    def test_str_representation(self, user):
        conv = GeneralConversation.objects.create(user=user, title="My Chat")
        assert str(conv) == "My Chat"
```

**`pytestmark = pytest.mark.django_db`** — tells pytest this test class needs a real database. pytest-django creates a temporary test database that's destroyed after the run.

### 2. Service Tests — Test business logic with mocks

```python
# apps/general_assistant/tests/test_services.py
from unittest.mock import patch, MagicMock
from general_assistant.services import ImageModalHandler

class TestQueryImageModel:
    @patch("general_assistant.services._hf_client")
    async def test_success(self, mock_client):
        # Arrange: set up the mock
        mock_result = MagicMock()
        mock_result.generated_text = "A dog running"
        mock_client.image_to_text.return_value = mock_result

        # Act: call the function
        handler = ImageModalHandler()
        result = await handler.query_image_model(b"fake-bytes")

        # Assert: check the result
        assert result == "A dog running"
        mock_client.image_to_text.assert_called_once_with(b"fake-bytes")

    @patch("general_assistant.services._hf_client")
    async def test_api_error_returns_fallback(self, mock_client):
        mock_client.image_to_text.side_effect = ConnectionError("DNS failed")

        handler = ImageModalHandler()
        result = await handler.query_image_model(b"fake-bytes")

        assert "unavailable" in result.lower()
```

**Mocking pattern:**
- `@patch("module.path.object")` — replaces the real object with a mock
- `.return_value` — what the mock returns when called
- `.side_effect` — raise an exception or return different values
- `.assert_called_once_with(...)` — verify it was called with specific args

### 3. API Tests — Test ViewSet endpoints

```python
from rest_framework.test import APIClient
from django.contrib.auth.models import User

@pytest.fixture
def api_client():
    user = User.objects.create_user(username='testuser', password='testpass')
    client = APIClient()
    client.force_authenticate(user=user)
    return client

class TestProductAPI:
    def test_list_products(self, api_client):
        response = api_client.get('/api/v1/products/')
        assert response.status_code == 200

    def test_create_product(self, api_client):
        response = api_client.post('/api/v1/products/', {
            'name': 'New Product',
            'price': '19.99',
            'category': 1,
        })
        assert response.status_code == 201

    def test_unauthenticated(self):
        client = APIClient()  # No auth
        response = client.get('/api/v1/orders/')
        assert response.status_code == 401
```

---

## The Arrange-Act-Assert Pattern

Every test follows this structure:

```python
def test_something():
    # ARRANGE — set up test data
    user = User.objects.create_user(username='test', password='pass')
    conversation = Conversation.objects.create(user=user)

    # ACT — do the thing you're testing
    message = Message.objects.create(
        conversation=conversation,
        content_type='TE',
        is_from_user=True
    )

    # ASSERT — check the result
    assert message.is_from_user is True
    assert message.conversation == conversation
```

---

## Pytest Fixtures

Fixtures are reusable test setup functions:

```python
# conftest.py (shared across all tests)
import pytest
from django.contrib.auth.models import User

@pytest.fixture
def user(db):
    return User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='testpass123'
    )

@pytest.fixture
def admin_user(db):
    return User.objects.create_superuser(
        username='admin',
        email='admin@example.com',
        password='adminpass123'
    )
```

Any test can use these by adding the fixture name as a parameter:

```python
def test_with_user(user):
    assert user.username == 'testuser'

def test_with_admin(admin_user):
    assert admin_user.is_staff is True
```

---

## Testing Async Code

This project uses `asyncio_mode = "auto"` — async tests just work:

```python
async def test_async_function():
    result = await some_async_function()
    assert result == expected

# With mocks:
@patch("module.function")
async def test_async_with_mock(mock_fn):
    mock_fn.return_value = "mocked"
    result = await function_under_test()
    assert result == "mocked"
```

---

## What to Test vs What NOT to Test

### Test:
- Your models (creation, constraints, methods)
- Your business logic (services, utilities)
- Your API endpoints (status codes, response shapes)
- Error handling (what happens when things fail)

### Don't test:
- Django's built-in functionality (ForeignKey works, save() works)
- Third-party libraries (LangChain's streaming works)
- Framework internals (DRF serialization works)

---

## Quick Exercise

1. Run the existing tests: `pytest apps/general_assistant/tests/ -v`
2. Read `test_models.py` and `test_services.py` — understand the mocking pattern
3. Write a new test for `VoiceModalHandler`:
   ```python
   class TestVoiceModalHandler:
       def test_init_creates_recognizer(self):
           handler = VoiceModalHandler()
           assert handler.recognizer is not None
   ```
4. Write a test for a model method using Arrange-Act-Assert
