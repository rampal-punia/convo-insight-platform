# ConvoInsight Frontend — Vision

> The product north star for the ConvoInsight web client. This document defines **what we are building, why it matters, and what "done" looks like** at the frontend layer. It is intentionally independent of any single release or contributor cohort.

---

## 1. Product Premise

ConvoInsight is a Customer Conversational Intelligence Platform. The backend already exposes a versioned REST API, real-time WebSocket channels, an NLP analysis pipeline (sentiment, intent, topic, NER), and a LangGraph-powered support agent with tool use.

The frontend is the **operator surface** that turns those capabilities into a product. It is the interface where:

- **Support agents and operations teams** review live conversations, intervene when needed, and act on AI-generated recommendations.
- **Analysts and managers** explore aggregated insights — sentiment trends, topic distributions, intent accuracy, agent performance.
- **Customers** interact with the AI support agent through a chat surface that streams responses, exposes tool actions, and degrades gracefully when the model is uncertain.
- **NLP practitioners** compare classification methods (fine-tuned BERT, few-shot GPT, RAG over pgvector) side by side in a reproducible playground.

The frontend exists to make the platform's intelligence **observable, actionable, and trustworthy**.

---

## 2. Vision Statement

> Build the most clear, fast, and inspectable conversational-intelligence cockpit on the web — one where every AI decision is explained, every metric is drillable, and every interaction feels real-time.

We are not building a generic admin panel. We are building a domain-specific cockpit for conversational AI, with three first-class concerns: **live conversations, model behaviour, and operational analytics**.

---

## 3. Strategic Pillars

The product is organised around five pillars. Every page, component, and interaction must serve at least one of them.

### Pillar 1 — Live Conversation Surface
A real-time chat experience for both end customers and operators. Streams tokens, surfaces tool calls (track order, modify order, cancel order, web search), shows intent and sentiment as they are detected, and supports human handoff.

### Pillar 2 — Insight Cockpit
Dashboards and analytics that turn raw conversation data into operational answers: which topics are trending, where sentiment is degrading, which intents the agent struggles with, how performance evolves week over week.

### Pillar 3 — NLP Playground
A side-by-side comparison surface for the three classification methods (fine-tuned BERT, few-shot GPT, RAG with pgvector) across the four NLP tasks (sentiment, intent, topic, NER). The playground is the platform's "show, don't tell" — the place that proves the science.

### Pillar 4 — Commerce Workbench
Order, product, and tracking management tied to the support agent's tool calls, so an operator can see the same data the agent acts on and override when required.

### Pillar 5 — Trust & Explainability
Every AI output is rendered with provenance: which method, which model, which confidence score, which tool calls fired, which documents were retrieved. No black boxes.

---

## 4. Non-Goals

To stay focused, the frontend explicitly **does not** aim to be:

- A general-purpose CRM or helpdesk replacement.
- A no-code workflow builder.
- A drag-and-drop dashboard editor.
- A mobile-native application (responsive web is in scope; native is not).
- A multi-tenant SaaS control plane (single-organisation deployments are the target).

These constraints are deliberate. They keep the product opinionated and the codebase small.

---

## 5. Experience Principles

These principles bind the design and implementation choices.

1. **Real-time by default.** If the backend can stream it, the UI streams it. Polling is a fallback, not a pattern.
2. **One screen, one job.** Each route has a single primary task. Secondary actions live behind clear affordances.
3. **Show the model's work.** Confidence scores, tool calls, retrieved context, and model identifiers are visible on every AI output.
4. **Operator-first density.** The product is used by professionals; information density beats whitespace.
5. **Keyboard-complete.** Every primary workflow is reachable without a mouse.
6. **Accessible from day one.** WCAG 2.2 AA is the floor, not a future milestone.
7. **Fast on a laptop, usable on a tablet.** Sub-second interactions on the dashboard; chat must remain responsive on slow networks.
8. **Honest empty, loading, and error states.** No spinners without context; no errors without a recovery path.

---

## 6. Target Users & Primary Jobs

| Persona | Primary Job-To-Be-Done | Key Surfaces |
|---|---|---|
| **End customer** | "Resolve my order issue without waiting for a human." | Customer chat, order tracking |
| **Support operator** | "Watch active conversations and step in when the agent struggles." | Live conversation queue, takeover panel |
| **Operations manager** | "Understand where conversations break and where the agent excels." | Insight dashboard, agent performance, recommendations |
| **NLP practitioner** | "Compare classification methods on real text and iterate on prompts/models." | Playground, evaluation views |
| **Administrator** | "Manage products, users, and platform configuration." | Admin workbench, product/order CRUD |

---

## 7. Twelve-Month North Star Outcomes

A successful frontend, twelve months from the first production deploy, demonstrably delivers:

1. **End-to-end customer chat** with streaming tokens, visible tool calls, and seamless handoff — measured by time-to-first-token < 1.5 s and zero dropped sessions on reconnect.
2. **Insight cockpit** with at least four production dashboards (conversations, sentiment, topics, agent performance), each backed by tested data hooks and exportable as CSV.
3. **Playground parity** with the backend's three classification methods and four NLP tasks, including saveable prompts and side-by-side result comparison.
4. **Operator handoff surface** that lets a human take over an in-flight agent conversation in under three seconds.
5. **WCAG 2.2 AA compliance** verified by automated and manual audits on every primary route.
6. **Performance budget enforced in CI**: Largest Contentful Paint < 2.0 s and Total Blocking Time < 200 ms on the dashboard route, measured on a mid-range laptop profile.
7. **Test coverage** of at least 80 % statements on `lib/` and at least 60 % on `app/` routes, with WebSocket flows covered by integration tests.
8. **Design system in place**: a documented component library (tokens, primitives, patterns) used across every route, with zero ad-hoc colour values outside the token set.

---

## 8. Success Metrics

The frontend's health is tracked along four metric families.

| Family | Indicator | Target |
|---|---|---|
| **Experience** | Time to first agent token (p95) | < 1.5 s |
| **Experience** | Interaction to next paint on dashboard (p95) | < 200 ms |
| **Reliability** | WebSocket reconnect success rate | > 99 % |
| **Reliability** | Unhandled promise rejections per session | 0 |
| **Quality** | Lighthouse Performance score on key routes | ≥ 90 |
| **Quality** | Lighthouse Accessibility score | 100 |
| **Adoption** | Weekly active operators using the live queue | Tracked, trend up |
| **Adoption** | Playground sessions per week | Tracked, trend up |

These targets are aspirations that drive engineering decisions; they are not gates for early releases.

---

## 9. Architectural Posture

The frontend is, and will remain:

- **A Next.js 16 App Router application written in JSX only.** Turbopack is the default builder. No TypeScript files are added to the repository.
- **Decoupled from the backend** behind the REST API and WebSocket protocol. No shared code, no server-rendered Django leakage into Next.js routes.
- **Single client codebase** — no separate mobile app, no separate admin app. Differentiation happens via routes and roles, not separate builds.
- **Server-rendered when the route is static or auth-agnostic, client-rendered when interactive.** Streaming and WebSockets force certain routes to be client components; everything else defaults to server.
- **Session-authenticated via Auth.js v5.** The Django JWT pair is held inside an encrypted, httpOnly Auth.js session cookie. `localStorage` is not used for tokens. Django remains the canonical identity store.

Anything that does not fit this posture requires a written design proposal before implementation.

---

## 10. What "Great" Looks Like

A new visitor opens the platform, signs in, and within thirty seconds can:

- Open a live customer chat and watch a response stream in token by token, with the active intent, sentiment, and tool calls visible on the side.
- Switch to the dashboard and see this week's sentiment trend, top three topics, and the agent's resolution rate.
- Jump to the playground, paste a sentence, run it through all three methods, and compare confidence scores in one view.
- Drill from any metric into the underlying conversations and read the exact messages that produced it.

Every one of those moments feels fast, deliberate, and explainable. That is the bar.
