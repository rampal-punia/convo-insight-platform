# ConvoInsight Frontend — Strategy

> How we get from the current Next.js skeleton to the product described in [VISION.md](./VISION.md). This document is project-specific: it names the architecture choices, the phased build plan, and the engineering disciplines that apply to every change.

This is a living document. When a decision in it changes, the document changes with it in the same pull request.

---

## 1. Strategic Context

The backend is mature: a versioned REST API (21 ViewSets), JWT auth, four WebSocket consumers, a LangGraph support agent, a three-method NLP pipeline, and PostgreSQL with pgvector. The frontend is a deliberate skeleton: a Next.js 15 App Router app with a landing page, a login page, a single authenticated `/products` page, and a thin `lib/api.js` client.

The strategy below is therefore **not** a greenfield plan. It is a build-out plan: turn the skeleton into the cockpit described in the vision, without rewriting what already works and without diverging from the backend's contracts.

---

## 2. Strategic Bets

We are making the following bets. Each one shapes the implementation.

| Bet | What it means in practice |
|---|---|
| **App Router + JSX, no TypeScript** | All new code is `.jsx` / `.js`. Type safety comes from JSDoc, schema validation at the API boundary, and tests — not from a compiler. |
| **One API client, one auth model** | Every network call goes through `lib/api.js`. WebSocket auth uses the same JWT, attached as a query parameter (matching the backend's `JWTAuthMiddleware`). |
| **Server components by default** | Routes are server components unless they need browser APIs, hooks, or WebSockets. Client boundaries are explicit and minimal. |
| **TailwindCSS + a token-driven design system** | All styling is utility-first. Colours, spacing, radii, and typography flow from `tailwind.config.js` tokens. No ad-hoc hex values in JSX. |
| **Composition over framework lock-in** | We prefer small, swappable libraries (`recharts` for charts, `zod` for schema validation, `swr` for cache + revalidation) over heavy frameworks. |
| **Tests are part of the feature** | Every page ships with at least one component test; every `lib/` function ships with unit tests. CI fails on coverage regressions in `lib/`. |

These bets are revisited only when a measured pain point forces a change. They are not revisited because of taste.

---

## 3. Target Architecture

```
frontend/
├── app/                        # Next.js routes (App Router)
│   ├── (public)/               # Marketing & auth-free routes
│   │   ├── page.jsx            # Landing
│   │   └── login/page.jsx      # Login
│   ├── (app)/                  # Authenticated shell
│   │   ├── layout.jsx          # Auth guard + chrome (nav, header)
│   │   ├── dashboard/          # Insight cockpit
│   │   ├── conversations/      # List + detail (live chat)
│   │   ├── chat/               # Customer-facing support chat
│   │   ├── playground/         # NLP playground (BERT / GPT / RAG)
│   │   ├── orders/             # Orders workbench
│   │   ├── products/           # Product catalogue
│   │   └── analytics/          # Sentiment, topics, performance
│   └── api/                    # Next.js route handlers (only if absolutely needed)
│
├── components/                 # Reusable JSX components
│   ├── ui/                     # Primitives (Button, Card, Input, Badge…)
│   ├── chat/                   # Message bubble, stream indicator, tool-call panel
│   ├── charts/                 # Recharts wrappers (line, bar, donut, sparkline)
│   └── layout/                 # Nav, Sidebar, PageHeader
│
├── lib/                        # Framework-agnostic logic
│   ├── api.js                  # REST client (today)
│   ├── ws.js                   # WebSocket client with reconnect + auth
│   ├── auth.js                 # Token storage, refresh, hooks
│   ├── hooks/                  # useApi, useWebSocket, useAuthGuard
│   ├── schemas/                # Zod schemas mirroring backend serializers
│   └── format.js               # Date, currency, number, status formatters
│
├── __tests__/                  # Mirrors app/ and lib/ structure
├── docs/                       # This folder — vision, strategy, status, contributing
├── public/                     # Static assets
└── …                           # Config files (next, tailwind, postcss, vitest)
```

This is the target. The current tree (see [CURRENT_STATUS.md](./CURRENT_STATUS.md)) is a strict subset and grows toward this layout one phase at a time.

---

## 4. Phased Roadmap

Each phase has a clear exit criterion. We do not move to the next phase until the exit criterion is met. Phases are ordered by dependency, not by team preference.

### Phase 0 — Foundations (current)
**Goal:** A trustworthy baseline: design tokens, shared layout, auth guard, API/WS clients, test harness.

- Define Tailwind tokens (colours, spacing scale, typography ramp, radii).
- Add a route-grouped layout: `(public)` and `(app)` segments with an auth guard in `(app)/layout.jsx`.
- Promote `lib/api.js` into a small `lib/` module: `api.js`, `auth.js`, `ws.js`, `hooks/useApi.js`.
- Add `zod` and define schemas for the auth and product responses as the reference pattern.
- Stand up a `components/ui/` primitive set: `Button`, `Card`, `Input`, `Badge`, `Spinner`, `EmptyState`, `ErrorState`.
- Add a navigation chrome (top nav + breadcrumb) used by every authenticated page.

**Exit criterion:** A new page can be added in under 20 lines of route code by composing the layout, the auth guard, the API hook, and the UI primitives. The current `/products` and `/login` pages are refactored onto this baseline with no behavioural change.

### Phase 1 — Operator Cockpit MVP
**Goal:** Operators can see conversations and live metrics.

- `/dashboard` — KPIs (total conversations, average sentiment, resolution rate, top intent) sourced from `/api/v1/conversation-metrics/`, `/api/v1/agent-performance/`, `/api/v1/topics/trending/`.
- `/conversations` — paginated list with status, intent, sentiment badge, last-message timestamp.
- `/conversations/[id]` — message timeline with role, content, sentiment score, intent, and any `tool_calls` payload from `ai_text`.
- `/orders` and `/orders/[id]` — list + detail with the tracking timeline.

**Exit criterion:** An operator can sign in, see today's activity on the dashboard, click into a conversation, and read the full transcript with AI metadata. Each page has loading, empty, error, and unauthorised states tested.

### Phase 2 — Real-Time Chat
**Goal:** Live, streaming chat for both customers and operators.

- `lib/ws.js` — typed (via JSDoc) wrapper around `WebSocket` with auto-reconnect, JWT attach, ping/pong, and event-typed callbacks.
- `/chat/[conversationId]` — customer-facing chat against `ws/support_agent/<id>/` with streaming tokens, tool-call indicators, and intent/sentiment side panel.
- `/conversations/[id]/live` — operator view that listens to the same conversation with a "take over" affordance (placeholder action until backend handoff lands).
- Reconnect strategy: exponential backoff up to 30 s, max 10 attempts, visible status pill.

**Exit criterion:** A customer can hold a multi-turn conversation with the support agent through the UI, sees tokens stream, sees tool calls (track/modify/cancel/search), and the connection recovers from a forced disconnect.

### Phase 3 — NLP Playground
**Goal:** Side-by-side comparison of the three classification methods across the four NLP tasks.

- `/playground` — text input, task selector (sentiment / intent / topic / NER), method selector (BERT / GPT / RAG), and a results panel that shows confidence scores per method.
- Wire to `ws/nlp_playground/` for streaming results; fall back to the REST `/api/v1/nlp/*` endpoints for non-streaming methods.
- Save and reload prompts in `localStorage` (server-side persistence is a later iteration).
- Render confidence as a bar with explicit numeric value; render RAG retrievals as an expandable list.

**Exit criterion:** A practitioner can paste a sentence, run all three methods on the same text, and read a clear comparison with provenance.

### Phase 4 — Analytics Depth
**Goal:** Drillable analytics on sentiment, topics, intents, and agent performance.

- `/analytics/sentiment` — time series of `overall_sentiment_score` with day/week/month granularity.
- `/analytics/topics` — distribution from `/api/v1/topic-distributions/` with click-through to conversations carrying a topic.
- `/analytics/intents` — confusion matrix style view from `/api/v1/intent-predictions/`.
- `/analytics/agents` — per-agent performance from `/api/v1/agent-performance/` with sparkline trend per metric.
- CSV export for every chart's underlying data.

**Exit criterion:** Every chart links to the conversations behind its data. No analytics page is a dead end.

### Phase 5 — Polish & Hardening
**Goal:** Production-grade quality.

- Enforce a performance budget in CI (Lighthouse on `/dashboard`, `/chat/[id]`, `/playground`).
- WCAG 2.2 AA audit on every authenticated route; fix to compliance.
- Add an end-to-end smoke test (Playwright) that signs in, opens a conversation, and runs a playground analysis.
- Token storage migration plan: document the path from `localStorage` to httpOnly cookies and gate it behind a backend change.
- Error monitoring hook (Sentry or equivalent) wired through `lib/api.js` and `lib/ws.js`.

**Exit criterion:** The application meets the metrics defined in `VISION.md` §8 on the primary routes.

---

## 5. Cross-Cutting Concerns

The following concerns apply to every phase and every change.

### 5.1 Authentication & Sessions
- JWT access (~5 min) + refresh (~1 day) lifetimes, set by the backend.
- Tokens live in `localStorage` today. The hardening path to httpOnly cookies is documented in Phase 5 and requires a coordinated backend change; it is not done piecemeal.
- A single auth guard lives in `(app)/layout.jsx`. No page implements its own guard. Pages that need the current user use a `useCurrentUser()` hook backed by `/api/v1/users/me/`.

### 5.2 Data Fetching
- Client-side reads use a single `useApi(path, options)` hook backed by `lib/api.js`. The hook returns `{ data, error, isLoading, refresh }`.
- Server components fetch directly via `lib/api.js` when no browser state is required.
- We will introduce SWR (or React Query) only when concrete duplication and revalidation needs justify it. Until then, the simple hook is enough.

### 5.3 Schema Validation
- Every API response that drives a UI decision is validated with a `zod` schema in `lib/schemas/`.
- A schema mismatch is reported as an explicit error, not silently rendered.

### 5.4 Real-Time
- All WebSocket access goes through `lib/ws.js`. Direct `new WebSocket(...)` calls in pages are forbidden.
- Every connection shows a visible state: connecting, connected, reconnecting, failed.

### 5.5 Styling & Design System
- All visual values come from Tailwind tokens. No literal colour values in JSX.
- A `components/ui/` primitive is preferred over a one-off `<div>` with classes when the primitive exists.
- Dark mode is on the roadmap; it is enabled by Tailwind's `class` strategy from day one to avoid retrofitting.

### 5.6 Accessibility
- Every interactive element has a visible focus state and an accessible name.
- Forms use `<label>` paired with `htmlFor`; errors are linked with `aria-describedby`.
- Colour is never the only signal for status; pair it with an icon or text.

### 5.7 Performance
- Routes default to server components. Client boundaries are added with intent and reviewed in PRs.
- Images go through `next/image`. Charts are loaded with `next/dynamic` to keep the dashboard bundle small.
- Avoid client-side libraries over 100 kB minified+gzip unless the value is proven.

### 5.8 Testing
- `vitest` + `@testing-library/react` for unit and component tests.
- Mock `lib/api.js` in page tests; test `lib/api.js` itself by stubbing `fetch`.
- WebSocket flows are tested with a stub server (a hand-rolled `EventTarget`-based mock).
- CI runs `npm test` on every pull request and blocks merge on failure.

### 5.9 Observability
- All errors from `lib/api.js` and `lib/ws.js` flow through a single error sink so a real monitoring backend can be plugged in once.
- Each WebSocket message type is logged at `debug` level behind a `NEXT_PUBLIC_LOG_LEVEL` flag.

### 5.10 Security
- Never log tokens, raw request bodies, or PII.
- Sanitise any HTML rendered from API data. The agent's `content` field is plain text; if rich rendering is added, it goes through a vetted Markdown renderer with sanitisation enabled.
- CSRF is not a concern for the JWT flow, but it will be re-evaluated when cookies replace `localStorage`.

---

## 6. Definition of Done (per change)

A pull request is ready for review when **all** of the following are true:

1. Lints clean (`npm run lint`).
2. Tests pass (`npm test`) and new behaviour has at least one test.
3. The change respects the route grouping and the design tokens; no ad-hoc colours.
4. Loading, empty, error, and unauthorised states are covered for any new data-fetching page.
5. Accessibility checklist applied: keyboard reachable, labelled controls, sufficient contrast, focus visible.
6. The PR description names the phase (§4) the change advances and the route(s) it touches.

---

## 7. Decision Log (lightweight ADRs)

We keep a single append-only file `frontend/docs/decisions.md` (created when the first decision is recorded). Each entry is short: title, date, context, decision, consequences. We log a decision when it changes the answer to "how should new code be written here?"

Examples of decisions worth recording: introducing SWR/React Query, replacing `localStorage` tokens with cookies, adopting a chart library, choosing a forms library, changing the route grouping.

---

## 8. Out of Scope (for the current strategy horizon)

To remove ambiguity, the following are explicitly **not** part of this strategy:

- TypeScript migration. JSX-only stands.
- A monorepo or shared package with the backend.
- A separate mobile or native client.
- A custom design system framework (we use Tailwind primitives, not a from-scratch CSS framework).
- Server-side persistence of playground prompts (later, behind a backend feature).
- Multi-tenant theming or white-labelling.

If any of these become necessary, they earn a decision-log entry and a revision of this strategy.
