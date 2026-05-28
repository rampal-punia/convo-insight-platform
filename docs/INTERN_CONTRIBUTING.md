# Intern Contributing Guide — ConvoInsight Platform

> A step-by-step guide to making your first contribution. From picking an issue to getting your PR merged.

**Prerequisite:** You've completed the [Intern Onboarding](INTERN_ONBOARDING.md) — the full stack runs, you can navigate the codebase, and tests pass locally.

---

## Table of Contents

1. [How Contributions Work Here](#how-contributions-work-here)
2. [Setting Up Your Git Workflow](#setting-up-your-git-workflow)
3. [Finding Your First Issue](#finding-your-first-issue)
4. [The Branch Workflow (Step by Step)](#the-branch-workflow-step-by-step)
5. [Writing the Code](#writing-the-code)
6. [Writing Tests](#writing-tests)
7. [Running Quality Checks](#running-quality-checks)
8. [Committing Your Changes](#committing-your-changes)
9. [Opening a Pull Request](#opening-a-pull-request)
10. [The Review Process](#the-review-process)
11. [After Your PR is Merged](#after-your-pr-is-merged)
12. [Common Mistakes & Fixes](#common-mistakes--fixes)
13. [Cheat Sheet](#cheat-sheet)

---

## How Contributions Work Here

```
You pick an issue → Create a branch → Write code + tests → Open PR → Review → Merged
```

We follow a **fork-based workflow** (if you're an external contributor) or a **branch-based workflow** (if you have push access). Ask your mentor which one applies to you.

### Branch structure:

```
master        ← stable, production-ready (never push directly here)
  └── development  ← integration branch (PRs target this)
       └── feature/your-feature  ← your branch
```

---

## Setting Up Your Git Workflow

### If you have push access (internal interns):

```bash
# Clone the repo (you only do this once)
git clone https://github.com/rampal-punia/convo-insight-platform.git
cd convo-insight-platform

# Set up the upstream remote (for staying in sync)
git remote add upstream https://github.com/rampal-punia/convo-insight-platform.git
```

### If you're contributing via fork (external):

```bash
# 1. Fork the repo on GitHub (click "Fork" button)

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/convo-insight-platform.git
cd convo-insight-platform

# 3. Add the original repo as "upstream"
git remote add upstream https://github.com/rampal-punia/convo-insight-platform.git

# 4. Verify your remotes
git remote -v
# origin    https://github.com/YOUR-USERNAME/convo-insight-platform.git (fetch)
# upstream  https://github.com/rampal-punia/convo-insight-platform.git (fetch)
```

### Configure Git (do this once):

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"
```

Make sure the email matches the one on your GitHub account, or your commits won't show up on your profile.

---

## Finding Your First Issue

### Labels to look for:

| Label | Meaning |
|-------|---------|
| `good-first-issue` | Designed for newcomers — start here |
| `help-wanted` | The team wants help with this |
| `bug` | Something is broken |
| `enhancement` | New feature or improvement |
| `documentation` | Doc improvements (great first contribution) |

### Where to find issues:

1. Go to the repo → **Issues** tab → filter by `good-first-issue`
2. Check the project board for "To Do" items
3. Ask your mentor — they'll assign you something appropriate

### Before you start:

1. **Check if someone is already working on it** — read the issue comments
2. **Comment on the issue** saying you'd like to work on it
3. **Wait for assignment** — a maintainer will assign it to you
4. If no one responds in 2 days, ping your mentor

---

## The Branch Workflow (Step by Step)

### 1. Sync with the latest code

Always start from the latest `development`:

```bash
cd convo-insight-platform
git checkout development
git fetch upstream
git merge upstream/development

# If using fork, also push to your fork:
git push origin development
```

### 2. Create your feature branch

```bash
# Branch naming conventions:
# feature/add-product-reviews    ← new feature
# fix/order-status-bug           ← bug fix
# docs/api-endpoints             ← documentation
# refactor/order-serializer      ← code cleanup

git checkout -b feature/your-feature-name development
```

**Good branch names:** `feature/add-search`, `fix/login-redirect-loop`, `docs/contributing-guide`
**Bad branch names:** `my-branch`, `fix`, `test`, `updates`

### 3. You're ready to code

You're now on your own branch. Any commits you make will stay here until you push and open a PR.

---

## Writing the Code

### Before you change anything, read the relevant code:

1. Find the relevant files (use `grep`, your IDE, or the lesson docs)
2. Read the existing patterns — follow them
3. Understand what's already there before adding to it

### Where things go:

| What you're adding | Where it goes |
|-------------------|---------------|
| New database table | `apps/<app>/models.py` |
| New API endpoint | `apps/api/v1/views_<app>.py` |
| New serializer | `apps/api/v1/serializers_<app>.py` |
| New URL | Already handled by the DRF Router (auto-generated) |
| Background task | `apps/<app>/tasks.py` |
| Test | `apps/<app>/tests/test_<what>.py` |
| Management command | `apps/<app>/management/commands/<name>.py` |
| Template | `backend/templates/<app>/` |
| Form | `apps/<app>/forms.py` |

### Code style rules:

- **Line length:** 120 characters (enforced by `ruff`)
- **Quotes:** Single quotes in Python (`'hello'`), double in JSX (`"hello"`)
- **Imports:** `from <app> import ...` (not `from apps.<app> import ...`)
- **Frontend:** JSX only — never create `.ts` or `.tsx` files
- **Indentation:** 4 spaces in Python, 2 spaces in JSX

### Patterns to follow:

```python
# Lazy imports for heavy ML libraries (torch, transformers, etc.)
def analyze(text):
    import torch  # Inside the function, not at the top of the file
    from transformers import AutoModel
    ...

# asyncio.to_thread for sync code in async context
result = await asyncio.to_thread(sync_function, args)

# Error handling for external APIs
try:
    result = await api_call()
except (ConnectionError, TimeoutError) as exc:
    logger.warning("API unavailable: %s", exc)
    return fallback_response
```

---

## Writing Tests

Every PR must include tests. No exceptions.

### Where to put tests:

```
apps/<app>/tests/
├── __init__.py
├── test_models.py       # Model creation, constraints, methods
├── test_views.py        # API endpoint tests
├── test_services.py     # Business logic, external API mocks
└── test_tasks.py        # Celery task tests
```

### The structure of a test:

```python
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.django_db  # Need database access


class TestMyFeature:
    """Tests for the new feature."""

    def test_happy_path(self):
        """The normal case — everything works."""
        # Arrange
        product = Product.objects.create(name="Widget", price=9.99)

        # Act
        result = product.apply_discount(10)

        # Assert
        assert result == 8.991

    def test_edge_case(self):
        """What happens at the boundary?"""
        product = Product.objects.create(name="Widget", price=0.01)
        result = product.apply_discount(10)
        assert result >= 0  # Price shouldn't go negative

    @patch("module.external_api_call")
    def test_external_api_failure(self, mock_api):
        """What if the API is down?"""
        mock_api.side_effect = ConnectionError("timeout")
        result = my_function()
        assert "unavailable" in result.lower()
```

### What to test:

| Scenario | Example |
|----------|---------|
| Happy path | Creating a product works |
| Edge cases | Zero price, empty name, max length |
| Error handling | API down, invalid input, missing data |
| Permissions | Unauthenticated user gets 401 |
| Authorization | Regular user can't access admin endpoints |

### Running your tests:

```bash
# Run all tests
pytest

# Run just your test file
pytest apps/products/tests/test_models.py -v

# Run one test class
pytest apps/products/tests/test_models.py::TestProduct -v

# Run one test method
pytest apps/products/tests/test_models.py::TestProduct::test_happy_path -v

# See print output
pytest -s apps/products/tests/test_models.py

# Run with coverage report
pytest --cov=apps/products --cov-report=term-missing
```

---

## Running Quality Checks

Before every commit, run these:

```bash
cd backend

# 1. Lint (finds code issues)
ruff check . --fix           # Auto-fix what it can
ruff check .                 # Show remaining issues

# 2. Format (makes code style consistent)
ruff format .
ruff format . --check        # Check without changing

# 3. Tests (verify nothing is broken)
pytest -v

# 4. Generate API schema (verify no schema errors)
python manage.py spectacular --file /tmp/schema.yml
```

### Or use the Makefile from the repo root:

```bash
cd convo-insight-platform
make lint     # runs ruff check + format
make test     # runs pytest
make check    # runs all of the above
```

**Rule:** All four checks must pass before you open a PR. If they don't, fix the issues first.

---

## Committing Your Changes

### What to commit:

```bash
# Stage specific files (preferred)
git add apps/products/models.py
git add apps/products/tests/test_models.py

# Or stage everything you changed (careful with this)
git add .
```

### What NOT to commit:

- `.env` files (contains secrets)
- `__pycache__/` directories
- `*.pyc` files
- `media/` uploads
- IDE config (`.vscode/`, `.idea/`)
- Large model files

### Writing commit messages:

```
<type>: <short description>

<optional longer explanation>
```

**Types:**

| Type | When to Use |
|------|------------|
| `Add` | New feature or file |
| `Fix` | Bug fix |
| `Update` | Changing existing behavior |
| `Refactor` | Code cleanup without behavior change |
| `Test` | Adding or updating tests |
| `Docs` | Documentation changes |

**Examples:**

```bash
git commit -m "Add: product search endpoint with filtering"
git commit -m "Fix: order total calculation for discounted items"
git commit -m "Test: add tests for OrderItem serializer validation"
git commit -m "Docs: update API docs for new search parameters"
```

**Good commit messages:**
- `Add: product search endpoint with name and category filtering`
- `Fix: WebSocket disconnect on HuggingFace API timeout`
- `Test: add 15 tests for Order model status transitions`

**Bad commit messages:**
- `fixed stuff`
- `updates`
- `WIP`
- `please work`

### Multiple commits are fine:

Don't try to squash everything into one giant commit. Small, focused commits are easier to review:

```bash
git commit -m "Add: ProductSearchSerializer with filter fields"
git commit -m "Add: search action on ProductViewSet"
git commit -m "Test: add tests for product search endpoint"
```

---

## Opening a Pull Request

### 1. Push your branch:

```bash
git push origin feature/your-feature-name
```

If using fork workflow and it's your first push to this branch:
```bash
git push -u origin feature/your-feature-name
```

### 2. Create the PR on GitHub:

1. Go to your repo (or fork) on GitHub
2. You'll see a yellow banner saying "feature/your-feature-name had recent pushes" — click **"Compare & pull request"**
3. **Base branch:** `development` (not `master`)
4. **Compare branch:** `feature/your-feature-name`

### 3. Write a good PR description:

```markdown
## What does this PR do?
Brief description of what you changed and why.

## Changes
- Added `GET /api/v1/products/search/` endpoint
- Added `ProductSearchSerializer` with name/category filtering
- Added 8 tests covering search, filters, and edge cases

## How to test
1. Start the server: `python manage.py runserver`
2. Get a JWT token via Swagger UI
3. Call `GET /api/v1/products/search/?name=widget`
4. Verify filtered results are returned

## Checklist
- [x] Tests added and passing
- [x] Lint passes (`ruff check .`)
- [x] Format passes (`ruff format . --check`)
- [ ] Schema clean (`python manage.py spectacular --file /tmp/schema.yml`)
```

### 4. Request reviewers:

- Assign your mentor as a reviewer
- If your PR touches multiple apps, request reviewers who know those areas

---

## The Review Process

### What to expect:

1. **Automated checks** run first (lint, tests, schema) — these must be green
2. A **reviewer reads your code** and leaves comments
3. You may get:
   - **Approval** — ready to merge
   - **Comments** — suggestions (optional to address)
   - **Changes requested** — you must fix these before merging

### How to handle review feedback:

```bash
# 1. Make the requested changes in your code
# 2. Commit and push
git add apps/products/views.py
git commit -m "Fix: address review feedback on search validation"
git push origin feature/your-feature-name

# 3. The PR updates automatically — no need to create a new one
# 4. Reply to each comment explaining what you changed
```

### Don't take feedback personally:

Code review is about making the code better, not judging you. Every developer — senior or junior — gets review feedback. It's how we learn from each other.

### Common review comments you'll see:

| Feedback | What It Means | What To Do |
|----------|--------------|------------|
| "Add a test for this" | Missing test coverage | Write a test covering the specific case |
| "Use select_related here" | N+1 query problem | Add `.select_related('user')` to the queryset |
| "This should be lazy imported" | Slow startup | Move the import inside the function |
| "Extract this to a method" | Code duplication | Create a helper method |
| "Add error handling" | Missing try/except | Wrap external calls in error handling |
| "Please fix lint" | Style issues | Run `ruff check . --fix && ruff format .` |

---

## After Your PR is Merged

```bash
# 1. Switch back to development
git checkout development

# 2. Pull the latest (including your merged changes)
git pull upstream development

# 3. Delete your feature branch (it's merged, no longer needed)
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name  # if pushed to remote

# 4. You're ready to start the next task!
```

### Keep your fork in sync (if using fork workflow):

```bash
git checkout development
git fetch upstream
git merge upstream/development
git push origin development
```

Do this **every time before you start a new branch** to avoid merge conflicts.

---

## Common Mistakes & Fixes

### Mistake: "I committed to the wrong branch"

```bash
# If you haven't pushed yet:
git stash                    # Save your changes
git checkout correct-branch  # Switch to the right branch
git stash pop                # Restore your changes
```

### Mistake: "I need to undo my last commit"

```bash
# Undo the commit but keep the changes:
git reset --soft HEAD~1

# Now re-commit with a better message or more changes
```

### Mistake: "I have merge conflicts"

```bash
# Don't panic. Open the conflicting files in your editor.
# Look for markers like:
# <<<<<<< HEAD
# your code
# =======
# their code
# >>>>>>> development

# Pick the right version (or combine both), then:
git add <resolved-file>
git commit -m "Fix: resolve merge conflicts"
```

### Mistake: "Tests pass locally but fail in CI"

Usually one of:
1. **Missing migration** — run `python manage.py makemigrations` and commit the migration file
2. **Different Python version** — check which Python version CI uses
3. **Missing test data** — your test depends on seed data that doesn't exist in CI

### Mistake: "I accidentally pushed .env"

```bash
# Tell your mentor IMMEDIATELY. Then:
# 1. Rotate all exposed API keys
# 2. Remove the file from git history:
git rm --cached .env
git commit -m "Fix: remove accidentally committed .env"
git push
```

### Mistake: "My branch is way behind development"

```bash
git checkout feature/your-feature-name
git fetch upstream
git rebase upstream/development

# If there are conflicts, resolve them one by one:
# git add <resolved-file>
# git rebase --continue

# After rebase, force push (this is the one time force push is OK):
git push origin feature/your-feature-name --force-with-lease
```

---

## Cheat Sheet

### Starting a new task:

```bash
git checkout development && git pull upstream development
git checkout -b feature/task-name development
# ... write code + tests ...
ruff check . --fix && ruff format . && pytest -v
git add <files>
git commit -m "Add: description of what you did"
git push origin feature/task-name
# → Open PR on GitHub targeting development
```

### Day-to-day commands:

```bash
# From backend/ with venv activated:
ruff check . --fix           # Lint and auto-fix
ruff format .                # Format code
pytest -v                    # Run tests
pytest <path> -v             # Run specific tests
python manage.py migrate     # Apply database changes
python manage.py shell       # Debug interactively
```

### Git quick reference:

```bash
git status                   # What changed?
git diff                     # See the changes
git log --oneline -10        # Recent commits
git stash                    # Temporarily save changes
git stash pop                # Restore saved changes
git branch -a                # List all branches
```

### When you're stuck:

1. **Read the error message** — it usually tells you exactly what's wrong
2. **Search the codebase** — `grep -r "pattern" apps/` or use your IDE
3. **Check the lesson docs** — `backend/docs/lessons/` has examples for everything
4. **Ask your mentor** — no question is too basic, especially in your first month
5. **Pair with another intern** — two brains are better than one

---

> Remember: your first PR doesn't need to be perfect. It needs to be tested, readable, and open to feedback. That's how every developer started. Good luck!
