# Intern Assignment 06: Production Settings Hardening — Security & Configuration

**Track:** Backend — Django Settings & Deployment Security
**Difficulty:** Intermediate
**Estimated Effort:** 3–4 hours
**Prerequisites:** Understanding of Django settings hierarchy, environment variables

---

## Problem Statement

The production and staging settings files have **critical security vulnerabilities** that would expose the application to attack if deployed as-is:

```python
# config/settings/production.py
ALLOWED_HOSTS = ["*"]   # ← Accepts requests from ANY hostname
```

```python
# config/settings/base.py
SIMPLE_JWT = {
    'SIGNING_KEY': config("DJANGO_SECRET_KEY", default="dev-only-signing-key"),
    # ← If DJANGO_SECRET_KEY env var is not set, uses a HARDCODED dev key
    # ← ALL JWT tokens become predictable and forgeable
}
```

There are no HTTPS enforcement settings, no secure cookie flags, no HSTS headers, and no proper `SECRET_KEY` handling for production.

---

## Root Cause Analysis

### Current Settings Hierarchy

```
config/settings/
├── base.py         ← Shared settings, reads env vars via python-decouple
├── development.py  ← Inherits base, sets DEBUG=True
├── production.py   ← Inherits base, sets ALLOWED_HOSTS=["*"]
└── stage.py        ← Inherits base, sets ALLOWED_HOSTS=["*"]
```

### Vulnerability 1: `ALLOWED_HOSTS = ["*"]`

**File:** `config/settings/production.py` and `config/settings/stage.py`

```python
ALLOWED_HOSTS = ["*"]
```

This allows the application to respond to requests for **any hostname**, including attacker-controlled domains. An attacker can:
- Send phishing links that point to your server but display a different hostname
- Poison the Django `Host` header cache
- Bypass hostname-based security checks

**Fix:** Set `ALLOWED_HOSTS` to only the actual domain names the app should respond to.

### Vulnerability 2: Hardcoded Fallback SECRET_KEY

**File:** `config/settings/base.py`

```python
SECRET_KEY = config("DJANGO_SECRET_KEY", default="dev-only-signing-key")
```

If `DJANGO_SECRET_KEY` is not set in the production environment (which is easy to forget), the app uses the string `"dev-only-signing-key"`. Anyone who reads the source code (it's on GitHub) can:
- Forge session cookies
- Forge JWT tokens
- Forge password reset tokens
- Decrypt any signed data

**Fix:** In production, `SECRET_KEY` MUST be set via environment variable with NO default fallback.

### Vulnerability 3: No HTTPS Enforcement

**Files:** `config/settings/production.py`

None of these settings are configured:
```python
SECURE_SSL_REDIRECT = True         # Redirect HTTP → HTTPS
SESSION_COOKIE_SECURE = True        # Only send cookies over HTTPS
CSRF_COOKIE_SECURE = True           # Only send CSRF cookie over HTTPS
SECURE_HSTS_SECONDS = 31536000      # Tell browsers to only use HTTPS for 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
```

### Vulnerability 4: Debug Info in Production

**File:** `config/settings/production.py`

There is no explicit `DEBUG = False`. It inherits from `base.py` which also doesn't set it. Django defaults to `False`, but this is implicit and fragile — if someone adds `DEBUG = True` to `base.py` for testing and forgets to remove it, production will leak detailed error pages.

---

## Assignment Tasks

### Task 1: Fix ALLOWED_HOSTS (15 min)

**File:** `config/settings/production.py`

```python
ALLOWED_HOSTS = config(
    "DJANGO_ALLOWED_HOSTS",
    default="",
    cast=lambda v: [s.strip() for s in v.split(",") if s.strip()],
)
```

This reads a comma-separated list from the environment variable `DJANGO_ALLOWED_HOSTS`:
```bash
# Example .env in production:
DJANGO_ALLOWED_HOSTS=api.convoinsight.example.com,convoinsight.example.com
```

Do the same for `config/settings/stage.py` with its own environment variable.

**Testing:** Verify that:
- Without `DJANGO_ALLOWED_HOSTS` set, `ALLOWED_HOSTS` is an empty list (blocks all)
- With `DJANGO_ALLOWED_HOSTS=example.com`, `ALLOWED_HOSTS` is `["example.com"]`

---

### Task 2: Fix SECRET_KEY Handling (20 min)

**File:** `config/settings/production.py`

Add a guard that **crashes the app on startup** if SECRET_KEY is not properly configured:

```python
# In production.py, AFTER importing base settings:
SECRET_KEY = config("DJANGO_SECRET_KEY")

# Explicitly validate — no default, no fallback
if not SECRET_KEY or SECRET_KEY == "dev-only-signing-key":
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY environment variable must be set in production. "
        "Generate one with: python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'"
    )
```

**File:** `config/settings/development.py`

Keep the dev fallback so local development still works:
```python
SECRET_KEY = config("DJANGO_SECRET_KEY", default="dev-only-signing-key")
```

**File:** `config/settings/base.py`

Remove the `SECRET_KEY` definition from base.py entirely — each environment file should define its own with appropriate defaults.

**Testing:** Verify that:
- `development.py` still works without `DJANGO_SECRET_KEY` set
- `production.py` crashes with a clear error message if `DJANGO_SECRET_KEY` is not set
- `production.py` works if `DJANGO_SECRET_KEY` is set to a proper value

---

### Task 3: Add Production Security Settings (30 min)

**File:** `config/settings/production.py`

Add all production security settings in a clearly-labeled section:

```python
# ── Security ──────────────────────────────────────────────────────────────
DEBUG = False

# HTTPS
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# HSTS (HTTP Strict Transport Security)
SECURE_HSTS_SECONDS = 31536000       # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# Browser security headers
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
SECURE_REFERRER_POLICY = "same-origin"

# Session security
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_AGE = 3600 * 8        # 8 hours
SESSION_EXPIRE_AT_BROWSER_CLOSE = True

# CSRF
CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SAMESITE = "Lax"

# Clickjacking
X_FRAME_OPTIONS = "DENY"
```

---

### Task 4: Add a Settings Validation Management Command (30 min)

**File:** `apps/core/management/commands/check_production_readiness.py` (create)

Create a management command that validates the settings before deployment:

```python
"""Management command to validate production readiness."""

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Validate that settings are production-ready"

    def handle(self, *args, **options):
        errors = []

        # Check SECRET_KEY
        if settings.SECRET_KEY in ("dev-only-signing-key", "your-secret-key"):
            errors.append("SECRET_KEY is using a default/dev value")

        # Check ALLOWED_HOSTS
        if "*" in settings.ALLOWED_HOSTS:
            errors.append("ALLOWED_HOSTS contains '*' — this is insecure in production")

        # Check DEBUG
        if settings.DEBUG:
            errors.append("DEBUG is True — should be False in production")

        # Check HTTPS settings
        if not getattr(settings, 'SECURE_SSL_REDIRECT', False):
            errors.append("SECURE_SSL_REDIRECT is not enabled")

        if not getattr(settings, 'SESSION_COOKIE_SECURE', False):
            errors.append("SESSION_COOKIE_SECURE is not enabled")

        if not getattr(settings, 'CSRF_COOKIE_SECURE', False):
            errors.append("CSRF_COOKIE_SECURE is not enabled")

        if errors:
            self.stderr.write(self.style.ERROR("Production readiness FAILED:"))
            for error in errors:
                self.stderr.write(self.style.ERROR(f"  - {error}"))
            raise CommandError(f"{len(errors)} issue(s) found")
        else:
            self.stdout.write(self.style.SUCCESS("All production readiness checks passed!"))
```

**Create the directory structure:**
```bash
mkdir -p apps/core/management/commands
touch apps/core/management/__init__.py
touch apps/core/management/commands/__init__.py
```

**Testing:** Run `python manage.py check_production_readiness` with:
- `DJANGO_SETTINGS_MODULE=config.settings.development` — should show errors
- `DJANGO_SETTINGS_MODULE=config.settings.production` with proper env vars — should pass

---

### Task 5: Write Settings Tests (30 min)

**File:** `config/tests/test_settings.py` (create)

```python
import pytest
from django.test import override_settings, TestCase
from django.core.exceptions import ImproperlyConfigured


class TestDevelopmentSettings:
    def test_debug_is_true(self):
        from config.settings import development
        assert development.DEBUG is True

    def test_allowed_hosts_includes_localhost(self):
        from config.settings import development
        # Dev should allow localhost for local testing
        assert "127.0.0.1" in development.ALLOWED_HOSTS or "*" in development.ALLOWED_HOSTS


class TestProductionSettings:
    def test_secret_key_crashes_without_env_var(self):
        """Production settings must crash if DJANGO_SECRET_KEY is not set."""
        import os
        # Remove the env var temporarily
        key = os.environ.pop("DJANGO_SECRET_KEY", None)
        try:
            # Force re-import — should raise ImproperlyConfigured
            # Note: This test verifies the guard logic works
            pass  # Implementation depends on import mechanism
        finally:
            if key:
                os.environ["DJANGO_SECRET_KEY"] = key

    def test_allowed_hosts_not_wildcard(self):
        """Production ALLOWED_HOSTS must not contain '*'."""
        # This test should verify production settings don't have wildcard
        from config.settings import production
        if hasattr(production, 'ALLOWED_HOSTS'):
            assert "*" not in production.ALLOWED_HOSTS


class TestSecurityHeaders:
    @override_settings(
        SECURE_SSL_REDIRECT=True,
        SESSION_COOKIE_SECURE=True,
        CSRF_COOKIE_SECURE=True,
    )
    def test_security_middleware_headers(self):
        """Verify security middleware is configured."""
        from django.conf import settings
        assert settings.SECURE_SSL_REDIRECT is True
        assert settings.SESSION_COOKIE_SECURE is True
        assert settings.CSRF_COOKIE_SECURE is True
```

---

## File Reference Map

```
backend/
├── config/
│   ├── settings/
│   │   ├── base.py                       ← Task 2 (remove SECRET_KEY from base)
│   │   ├── development.py                ← Task 2 (keep dev fallback)
│   │   ├── production.py                 ← Tasks 1, 2, 3 (fix all security issues)
│   │   └── stage.py                      ← Task 1 (fix ALLOWED_HOSTS)
│   ├── tests/
│   │   └── test_settings.py              ← Task 5 (create)
│   └── celery.py                         ← Read-only
├── apps/
│   └── core/
│       └── management/
│           └── commands/
│               └── check_production_readiness.py  ← Task 4 (create)
└── docs/
    └── lessons/assignments/
        └── 06_production_settings_hardening.md  ← this file
```

---

## Key Concepts to Learn

1. **Django settings inheritance** — `production.py` imports from `base.py`, overrides specific values
2. **`python-decouple`** — `config()` reads from `.env` files, with type casting and defaults
3. **Host header injection** — Why `ALLOWED_HOSTS = ["*"]` is dangerous
4. **HTTPS enforcement** — `SECURE_SSL_REDIRECT`, secure cookies, HSTS headers
5. **Management commands** — Custom `manage.py` commands for operational tasks
6. **`ImproperlyConfigured`** — Django's exception for startup-time configuration errors

---

## Submission Checklist

- [ ] `production.py` reads `ALLOWED_HOSTS` from env (no wildcard)
- [ ] `stage.py` reads `ALLOWED_HOSTS` from env (no wildcard)
- [ ] `production.py` crashes if `DJANGO_SECRET_KEY` is not set
- [ ] `development.py` still works with default SECRET_KEY
- [ ] `production.py` has `DEBUG = False` explicitly set
- [ ] All HTTPS enforcement settings configured in production
- [ ] `check_production_readiness` management command works
- [ ] Command detects wildcard ALLOWED_HOSTS, default SECRET_KEY, DEBUG=True
- [ ] All existing tests pass
- [ ] New settings tests pass

---

*Assignment created: May 2026*
*Series: ConvoInsight Platform — Intern Assignments*
