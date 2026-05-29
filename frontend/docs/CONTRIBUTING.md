# ConvoInsight Frontend — Contributing Guide (Interns)

> A practical, opinionated guide for interns and new contributors working on the Next.js frontend. Read it before your first pull request. Follow it on every pull request.

This guide is **frontend-specific**. For the broader project workflow (branch model, fork & sync, PR etiquette) see the repository-level [CONTRIBUTING.md](../../CONTRIBUTING.md). For the product context, read [VISION.md](./VISION.md), [STRATEGY.md](./STRATEGY.md), and [CURRENT_STATUS.md](./CURRENT_STATUS.md) — in that order.

---

## 1. Prerequisites

Before you touch the frontend, you should have:

- The backend running locally (`python manage.py runserver` on port 8000).
- The seeded demo data loaded (`python manage.py seed_demo`).
- Node.js 20 or newer installed (`node --version`) — Next.js 16 requires Node ≥ 20.
- A working sign-in to the Swagger UI at `http://localhost:8000/api/docs/`.

If any of those is missing, complete the [Intern Onboarding](../../docs/INTERN_ONBOARDING.md) first.

---

## 2. Local Setup

```bash
cd frontend
npm install
cp .env.example .env.local   # if you need to override the backend URL
npm run dev
```

Visit `http://localhost:3000`. Sign in with `demo_user_01` / `demo12345`. You should land on `/products`.

Useful scripts:

| Command | What it does |
|---|---|
| `npm run dev` | Start the dev server on port 3000 |
| `npm run build` | Production build |
| `npm run start` | Run the production build |
| `npm run lint` | ESLint (Next config) |
| `npm test` | Vitest, single run |
| `npm run test:watch` | Vitest, watch mode |
| `npm run test:coverage` | Vitest with coverage report |

---

## 3. Non-Negotiables

These rules are not preferences. They are the contract for every change.

1. **JSX only.** Do not add `.ts` or `.tsx` files. Type the boundaries with `zod` (when present) and JSDoc.
2. **Use the shared session.** Server code uses `auth()`; client code uses `useSession()`. Never store tokens in `localStorage` or `sessionStorage`. Auth.js wiring is documented in [AUTHJS_INTEGRATION.md](./AUTHJS_INTEGRATION.md).
3. **Use the shared API client.** Every HTTP call goes through `lib/api.js`. Never call `fetch` directly from a page.
4. **Use the shared WebSocket client** (when it lands in Phase 2). Never call `new WebSocket(...)` directly from a page.
5. **Respect the route grouping.** Auth-required pages live under `app/(app)/`. Public pages live under `app/(public)/`. Do not implement per-page auth checks once the shared layout exists.
6. **All new user-visible features land in the cockpit.** The Django templates tier (auth, staff dashboard) is **frozen** — existing pages are edited in place, no new routes are added there. See [backend/docs/SURFACE_TIERS.md](../../backend/docs/SURFACE_TIERS.md) for the full rules.
7. **No ad-hoc colours.** Use Tailwind tokens (`slate-*`, the project's defined palette). If you find yourself reaching for a hex value, raise it in the PR — we either extend the token set or pick the closest existing one.
8. **Loading, empty, error states are mandatory.** A data-fetching page without all three is not done.
9. **Every interactive control must be keyboard-reachable and labelled.** Inputs paired with `<label htmlFor>`, buttons with text or `aria-label`.
10. **No `console.log` in shipped code.** Use a single log helper (or remove the log) before opening the PR.
11. **Every PR includes tests** for the behaviour it changes. PRs that only touch styling can skip new tests but must still pass the suite.
12. **Single quotes, 2-space indent, trailing commas where ESLint allows.** Let the linter sort it; do not fight it.

---

## 4. Project Map (where things live)

```
frontend/
├── app/                    # Routes (Next.js App Router, JSX only)
├── components/             # Shared JSX components (introduced in Phase 0)
│   ├── ui/                 # Primitives: Button, Card, Input, Badge, …
│   ├── chat/               # Chat-specific pieces
│   ├── charts/             # Recharts wrappers
│   └── layout/             # Nav, Sidebar, PageHeader
├── lib/                    # Framework-agnostic helpers
│   ├── api.js              # REST client (today)
│   ├── ws.js               # WebSocket client (Phase 2)
│   ├── auth.js             # Token + user helpers (Phase 0)
│   ├── hooks/              # useApi, useWebSocket, useAuthGuard
│   └── schemas/            # zod schemas mirroring backend serializers
├── __tests__/              # Mirrors app/ and lib/
└── docs/                   # Vision, strategy, status, this guide
```

If you are unsure where a new file belongs, read the import. If it is React-aware, it goes under `components/` or `app/`. If it is framework-agnostic, it goes under `lib/`.

---

## 5. Picking a Task

There are three ways to pick a task, in order of preference:

1. **A `good-first-issue` ticket on GitHub.** These are scoped, reviewed, and ready to start.
2. **An item from the active phase checklist** in [CURRENT_STATUS.md](./CURRENT_STATUS.md) §11. Comment on the issue (or open one) before starting.
3. **A small, obvious improvement** you spotted while reading the code — typos, dead imports, missing accessible labels. These do not need an issue; they need a clean PR.

Before you write code, post a short comment on the issue:

> "Hi, I'd like to pick this up. My plan: 1) add `components/ui/Card.jsx` with variants A/B; 2) replace the four hand-rolled cards in `/products`; 3) add a test. ETA: a few days."

This is not bureaucracy. It prevents two interns from racing on the same task and it lets the maintainers correct your direction early.

---

## 6. Writing a Page (the recipe)

Every authenticated, data-fetching page follows the same shape. Copy it; do not invent a new shape.

```jsx
'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { api, getAccess } from '@/lib/api';

export default function ExamplePage() {
  const router = useRouter();
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!getAccess()) {
      router.push('/login');
      return;
    }
    api('/api/v1/example/')
      .then((res) => {
        setData(res.results ?? res);
        setLoading(false);
      })
      .catch((err) => {
        if (err.status === 401) {
          router.push('/login');
          return;
        }
        setError(err.data?.detail || err.message);
        setLoading(false);
      });
  }, [router]);

  if (loading) return <p className="p-8 text-slate-500">Loading…</p>;
  if (error) return <p className="p-8 text-red-600">{error}</p>;
  if (!data || data.length === 0) {
    return <p className="p-8 text-slate-500">Nothing to show yet.</p>;
  }

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="text-2xl font-bold">Example</h1>
      {/* render data */}
    </main>
  );
}
```

Once the shared `(app)` layout and `useApi` hook land, this recipe collapses further — keep an eye on [FRONTEND_GUIDE.md](./FRONTEND_GUIDE.md) for the current canonical version.

---

## 7. Writing a Test

We use Vitest with `@testing-library/react`. Mirror the source path under `__tests__/`.

```jsx
// __tests__/app/example.test.jsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import ExamplePage from '@/app/example/page';

vi.mock('@/lib/api', () => ({
  getAccess: vi.fn(() => 'mock-token'),
  api: vi.fn(() => Promise.resolve({ results: [{ id: 1, name: 'Alpha' }] })),
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

describe('ExamplePage', () => {
  it('renders fetched items', async () => {
    render(<ExamplePage />);
    await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument());
  });
});
```

Test what the user sees, not the implementation detail. Prefer `findByText` and `getByRole` over CSS selectors.

---

## 8. Talking to the Backend

The backend already exposes everything you need.

- **REST**: 21 ViewSets under `/api/v1/`. See `http://localhost:8000/api/docs/` (Swagger) for the live, authoritative schema.
- **WebSockets**:
  - `ws://localhost:8000/ws/chat/<conversation_id>/`
  - `ws://localhost:8000/ws/support_agent/<conversation_id>/`
  - `ws://localhost:8000/ws/order_support/<conversation_id>/`
  - `ws://localhost:8000/ws/general_assistant/<conversation_id>/`
  - `ws://localhost:8000/ws/nlp_playground/`

WebSocket auth attaches the JWT as a `?token=...` query parameter. The backend middleware validates it.

If you need a new endpoint or a new field on an existing one, do **not** wedge it into the frontend with a hack. Open an issue describing the contract you need and tag a backend maintainer. The frontend and backend evolve in lockstep through explicit contract changes.

---

## 9. Code Style Checklist (before you push)

Run through this list every time. It takes two minutes and saves a review round.

- [ ] `npm run lint` is clean.
- [ ] `npm test` is green.
- [ ] No `console.log` left behind.
- [ ] No commented-out code.
- [ ] No new direct `fetch` or `new WebSocket(...)` calls in pages.
- [ ] Loading, empty, error states are handled if the page fetches data.
- [ ] Every input has a paired label.
- [ ] Every button has accessible text.
- [ ] Tailwind classes use the existing palette; no raw hex values.
- [ ] No `.ts` or `.tsx` files were added.
- [ ] The PR description names the issue, the route(s) touched, and (if relevant) the strategy phase.

---

## 10. Pull Request Etiquette

1. **Branch from `development`.** Never from `master`.
2. **One PR, one purpose.** If you find yourself writing "and also…" in the description, split the PR.
3. **Keep it small.** Reviewable PRs are under 400 changed lines. Larger ones need a short design note in the description.
4. **Write the description for a future you.** Why the change, what it touches, how to test it manually, screenshots or short clips for UI changes.
5. **Respond to review comments quickly and explicitly.** "Done" with the commit SHA, or "deferring to a follow-up issue #N because …". Silence is the slowest path to a merge.
6. **Rebase, do not merge `development` into your branch repeatedly.** Squash-merge is the project default.

---

## 11. When You Get Stuck

In order:

1. Re-read the relevant section of [FRONTEND_GUIDE.md](./FRONTEND_GUIDE.md) and [STRATEGY.md](./STRATEGY.md). Most "how do I" questions are answered there.
2. Search closed issues and pull requests on GitHub for the same symptom.
3. Reproduce the issue with the smallest possible code change and paste that snippet — not a screenshot — when asking for help.
4. Ask in the project chat. Be specific: what you tried, what you expected, what happened, the exact error message.

Asking for help early is good. Asking for help instead of reading the code is not.

---

## 12. Growth Path

The frontend codebase is small on purpose. As an intern, you can credibly own a meaningful slice within a few weeks if you focus.

A sensible progression:

1. Land a small UI fix or a missing test (week 1).
2. Land one `components/ui/` primitive with tests and adopt it in one page (week 2).
3. Land a new authenticated page following the recipe in §6, with tests (week 3).
4. Pick one Phase 1 or Phase 2 item from [STRATEGY.md](./STRATEGY.md) and drive it to completion across multiple PRs (week 4 onward).

The goal is not to ship the most code. The goal is to ship code that the next intern will read and understand without asking you.

---

_Welcome aboard. Write code you would be proud to see on the project's homepage — because it eventually will be._
