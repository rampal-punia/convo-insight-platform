# ConvoInsight Frontend — Current Status

> A factual snapshot of where the frontend stands today, based on the code in this repository. This is the reference point that [STRATEGY.md](./STRATEGY.md) builds from. Update this document whenever a phase exit criterion is met or a structural change lands.

---

## 1. At a Glance

| Area | State |
|---|---|
| Framework | Next.js 15.0.3 (App Router) |
| Language | JSX only — no TypeScript files |
| UI library | React 19.0.0 |
| Styling | TailwindCSS 3.4.1 (default config, no custom tokens yet) |
| Testing | Vitest 4.1.7 + Testing Library + jsdom |
| API client | Single file at [frontend/lib/api.js](../lib/api.js) |
| Auth | JWT (access + refresh) stored in `localStorage` |
| Routes shipped | `/`, `/login`, `/products` |
| WebSocket support | Not yet wired on the frontend |
| Design system | Not yet established — Tailwind utilities used ad hoc |
| Strategy phase | **Phase 0 — Foundations** (in progress) |

---

## 2. Repository Layout (frontend/)

```
frontend/
├── app/
│   ├── globals.css           # Tailwind directives
│   ├── layout.jsx            # Minimal root layout (html/body, no nav)
│   ├── page.jsx              # Landing page with two link cards
│   ├── login/page.jsx        # Client component, login form
│   └── products/page.jsx     # Client component, authenticated list
├── lib/
│   └── api.js                # JWT-aware fetch wrapper
├── __tests__/
│   ├── app/                  # Tests for login and products pages
│   └── lib/                  # Tests for the API client
├── docs/
│   ├── FRONTEND_GUIDE.md     # Developer reference (architecture, patterns)
│   ├── VISION.md             # Product north star
│   ├── STRATEGY.md           # Build-out plan
│   ├── CURRENT_STATUS.md     # This document
│   └── CONTRIBUTING.md       # Intern-focused contribution guide
├── package.json
├── next.config.js            # Empty default config
├── tailwind.config.js        # Default content paths, no custom theme
├── postcss.config.js
├── vitest.config.js
├── vitest.setup.js
└── jsconfig.json             # @/* path alias
```

Note: there is no `components/` directory yet. All JSX lives in route files. Establishing `components/ui/` is a Phase 0 task.

---

## 3. Routes

| Route | File | Type | Auth Guard | Backend Endpoints Used |
|---|---|---|---|---|
| `/` | `app/page.jsx` | Server | — | None (links only) |
| `/login` | `app/login/page.jsx` | Client | — | `POST /api/v1/auth/token/` |
| `/products` | `app/products/page.jsx` | Client | Inline `getAccess()` check | `GET /api/v1/products/`, `POST /api/v1/auth/token/blacklist/` |

Observations:
- The auth guard is duplicated per page (currently only on `/products`). A shared `(app)` layout is not yet introduced.
- Logout is implemented on `/products` but not exposed from a shared nav (there is no shared nav).
- The landing page hardcodes a link to `/api/docs/` on the backend, which is correct for development but should be gated behind an environment flag for non-dev builds.

---

## 4. API Client (`lib/api.js`)

A single thin wrapper around `fetch`. Capabilities:

- Attaches `Authorization: Bearer <access>` from `localStorage` when `auth: true` (default).
- On `401`, attempts a single refresh via `/api/v1/auth/token/refresh/` and retries the original request once.
- Serialises JSON request bodies and parses JSON responses.
- Throws an `Error` with `status` and `data` properties on non-OK responses.
- Exposes `login`, `logout`, `me`, `getAccess`, `getRefresh`, `setTokens`, `clearTokens`.

Known gaps (tracked under Phase 0):

- No request abort / cancellation support.
- No retry policy beyond the single refresh on `401`.
- No central error sink for observability.
- No schema validation on responses.
- Token storage uses `localStorage`; migration to httpOnly cookies is in Phase 5.

---

## 5. WebSocket Coverage

The backend exposes four WebSocket consumers:

| Backend route | Consumer | Purpose |
|---|---|---|
| `ws/chat/<conversation_id>/` | `convochat.ChatConsumer` | Core conversation consumer |
| `ws/support_agent/<conversation_id>/` | `support_agent.SupportAgentConsumer` | LangGraph agent with tool calls |
| `ws/order_support/<conversation_id>/` | `orders.OrderSupportConsumer` | Order-scoped support |
| `ws/general_assistant/<conversation_id>/` | `general_assistant.GeneralChatConsumer` | Multimodal assistant |
| `ws/nlp_playground/` | `playground.NLPPlaygroundConsumer` | Streaming NLP playground |

The frontend currently consumes **none** of these. A `lib/ws.js` client and the first consumer surface (`/chat/[id]`) land in Phase 2.

---

## 6. Tests

Test files exist under `__tests__/` for the API client and the two real pages. They run via `npm test` (Vitest in jsdom mode).

- `__tests__/lib/api.test.js` — exercises the `fetch` wrapper.
- `__tests__/app/login.test.jsx` — exercises the login form.
- `__tests__/app/products.test.jsx` — exercises the products list with mocked `lib/api`.

There is no end-to-end test, no coverage gate in CI, and no Lighthouse budget yet. These are Phase 5 commitments.

---

## 7. Styling

- TailwindCSS is installed with the default theme.
- `tailwind.config.js` extends nothing; there are no custom design tokens.
- JSX uses utility classes directly. The colour vocabulary today: `slate-*`, `blue-600`, `red-50`/`red-600`, `green-100`/`green-800`, `yellow-100`/`yellow-800`.
- No icon set, no typography ramp, no dark-mode strategy yet.

Establishing the design token set (colours, spacing scale, radii, typography) and the `components/ui/` primitives is the central Phase 0 deliverable.

---

## 8. Tooling & Configuration

| Tool | Status |
|---|---|
| ESLint | Configured via `eslint-config-next` |
| Prettier | Not configured (Next defaults apply through ESLint) |
| Husky / pre-commit | Not configured |
| CI | Backend pipeline exists; frontend lint/test in CI is not yet wired |
| Bundle analysis | Not yet wired |
| Lighthouse / perf budget | Not yet wired |

---

## 9. Environment & Configuration

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Django backend base URL |

There is no environment-specific config file beyond `.env.example`. Production environment variables will be documented as part of the deployment work in Phase 5.

---

## 10. What Works End-to-End Today

A new contributor, after the standard backend setup, can:

1. Start the backend (`python manage.py runserver`).
2. Start the frontend (`npm install && npm run dev`).
3. Visit `http://localhost:3000`.
4. Sign in with seeded credentials (`demo_user_01` / `demo12345`).
5. See the product list rendered from `/api/v1/products/`, with a working logout.
6. Run the test suite with `npm test`.

That is the full surface today. Everything beyond this is on the roadmap in [STRATEGY.md](./STRATEGY.md).

---

## 11. Active Phase Checklist (Phase 0 — Foundations)

- [ ] Tailwind design tokens defined in `tailwind.config.js` (colours, spacing, typography, radii).
- [ ] Route groups introduced: `(public)` and `(app)`.
- [ ] Shared `(app)/layout.jsx` with auth guard and top navigation.
- [ ] `lib/` split into `api.js`, `auth.js`, `ws.js`, `hooks/useApi.js`.
- [ ] `components/ui/` primitives shipped: `Button`, `Card`, `Input`, `Badge`, `Spinner`, `EmptyState`, `ErrorState`.
- [ ] `zod` introduced; schemas for `LoginResponse`, `Product`, `User` added.
- [ ] `/products` and `/login` refactored onto the new baseline with no behavioural change.
- [ ] Frontend lint + test added to CI.

Once every item is checked, this document is updated and the strategy phase advances to Phase 1.

---

_Last updated: keep this in sync with reality. If you change something structural, update the same PR that changes the code._
