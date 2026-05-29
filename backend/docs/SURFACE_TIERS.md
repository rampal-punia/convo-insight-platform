# Surface Tiers — Why ConvoInsight Ships Both Django Templates and a Next.js Cockpit

> A first-principles explanation of the platform's three frontend tiers and the rules that keep them from colliding. Read this **before** you decide where a new page belongs.

For the product context this sits inside, see [backend/docs/VISION.md](../VISION.md) and [frontend/docs/VISION.md](../../../frontend/docs/VISION.md). For the build-out plans, see [frontend/docs/STRATEGY.md](../../../frontend/docs/STRATEGY.md) and the README roadmap.

---

## 1. The Question

ConvoInsight ships two visible web tiers in the same repository:

- A **server-rendered Django tier** built with Bootstrap 5 and `django-crispy-forms` (`backend/templates/`).
- A **client-rendered Next.js 15 tier** built with the App Router, JSX, and TailwindCSS (`frontend/`).

A reasonable reviewer will ask: *why both?* This document is the answer. It is not a retrospective justification; it is the design.

---

## 2. The Three Tiers (not two)

The platform exposes **three** surfaces, each with a different audience, a different implementation, and a different remit. Treating them as three — not as "Django vs Next.js" — removes the false dichotomy.

| Tier | Implementation | Audience | Primary remit |
|---|---|---|---|
| **Admin** | Django admin (`/admin/`) | Engineers, data ops | Inspect and edit any model, debug, run side effects. |
| **Staff / Operator (server-rendered)** | Django templates + Bootstrap 5 + crispy-forms (`backend/templates/`) | Internal users, staff, content authors | Auth (login, signup, password reset, profile), staff dashboards, internal CRUD that benefits from Django forms and the messages framework, server-rendered iteration surfaces. |
| **Product Cockpit (client-rendered)** | Next.js 15 App Router + Tailwind (`frontend/`) | End customers, operators using the product | Customer-facing chat, operator live-queue, NLP playground comparison view, analytics cockpit, anything streaming, anything shipped to an end customer. |

Every feature lives in exactly one tier. The tier is chosen once, recorded in the pull request, and not revisited without a written proposal.

---

## 3. Why This Is the Right Design (not an accident)

Five reasons make this the deliberate choice for a platform of this shape.

### 3.1 Each tier plays to its framework's strengths
Django ships solved primitives for the things that are tedious to rebuild: a battle-tested auth flow, form rendering with validation and CSRF, the messages framework, the admin, and an ORM-first templating model. The cockpit, by contrast, needs streaming WebSocket UIs, optimistic updates, route-grouped layouts, and a sharp visual identity — all things the Next.js App Router does naturally. Using each framework where it excels is faster than forcing one to do both jobs.

### 3.2 Operational resilience
Because auth and the staff dashboard are server-rendered, the platform is **demonstrably usable end-to-end even when the cockpit is mid-rebuild**. Branches that touch the Next.js layer cannot brick a live demo. That property is worth a lot during heavy iteration.

### 3.3 Onboarding ramp without language gymnastics
A backend contributor can ship a real, visible feature in the staff tier on their first week without learning Tailwind, JWT storage, the App Router, or React 19. A frontend contributor can ship a cockpit route the same week without touching Python. Both ramps run in parallel without a hand-off bottleneck.

### 3.4 Two patterns, one domain — a rare teaching artefact
Engineers who see the same domain delivered both as a server-rendered MPA and as an SPA-over-API learn the trade-offs viscerally: page-by-page navigation versus client routing, form posts versus JSON requests, session cookies versus JWTs, server-rendered HTML versus streamed tokens. ConvoInsight makes both patterns inspectable side by side.

### 3.5 A real production posture
Production Django systems almost always run an internal server-rendered surface alongside an external SPA/API surface. The Django admin alone is not enough; it is too powerful for non-engineers and too coarse for daily staff workflows. Codifying a middle tier — the templates tier — matches real-world deployments rather than the simplified "API-only" textbook diagram.

---

## 4. Why "Both" Goes Wrong If Left Undisciplined

The benefits above only land if the tiers do not bleed into each other. Without rules, the failure modes are predictable:

- **Style drift** between Bootstrap and Tailwind that ends with two visual languages.
- **Feature duplication** — the same dashboard built once on each tier, with different UX.
- **Decision paralysis** for contributors: "where should this page live?"
- **Doubled maintenance** — two test stacks, two deploy concerns, two sets of bugs for one feature.

The remit table in §2 plus the rules in §5 exist specifically to prevent these failure modes.

---

## 5. The Rules

The rules are short, enforceable in review, and never optional.

**R1. One tier per feature.** A feature implemented in one tier is not re-implemented in another without a written proposal that justifies the cost.

**R2. The contract is the boundary.** The cockpit talks to the backend through `/api/v1/*` and `ws/*` only. It does not import from `backend/`. The staff templates talk to models and services directly; they do not call `/api/v1/`.

**R3. Auth belongs to the staff tier.** Login, signup, password reset, and profile management are delivered as Django views with crispy forms. The cockpit assumes a JWT exists and provides a login screen that POSTs to `/api/v1/auth/token/`; it does not own password reset flows or social-login forms.

**R4. Streaming and real-time belong to the cockpit.** Anything that needs WebSocket UIs, token streaming, or sub-second optimistic updates is built in the Next.js layer. The staff tier may show static summaries of conversations but does not render live chat.

**R5. Admin belongs to Django admin.** Daily power-user data ops live in `/admin/`. The staff tier does not re-implement admin features; it complements admin with role-aware workflows.

**R6. PR description names the tier.** Every PR that adds or moves a page answers "which tier and why?" — one sentence is enough. Reviewers reject PRs that build the wrong tier.

**R7. Cross-tier visual consistency is brand-level, not pixel-level.** Both tiers share the project's name, palette, and core typography choices. They do **not** share components; each uses its framework's idioms (Bootstrap utilities in templates, Tailwind tokens in the cockpit).

**R8. Deprecation requires a written proposal.** If at some future point the cockpit absorbs a staff workflow, the corresponding template is removed in the same PR after the cockpit version is live and tested. No silent parallel maintenance.

---

## 6. Decision Cheatsheet

When you cannot tell which tier a new feature belongs to, walk the table top-to-bottom. The first row that matches wins.

| Characteristic | Tier |
|---|---|
| Needs an HTML form posted with CSRF, validation, and Django messages | Staff templates |
| Needs WebSocket streaming, optimistic updates, or sub-second feedback | Cockpit |
| Is the auth / account / password reset surface | Staff templates |
| Is the support chat or operator live-queue | Cockpit |
| Is a quick internal dashboard used by 1–10 staff | Staff templates |
| Is a polished dashboard a customer or external user will see | Cockpit |
| Touches a model that has a Django admin and you need bulk ops | Django admin |
| Is the NLP playground comparison view (BERT vs GPT vs RAG side by side) | Cockpit |
| Is a one-off form to trigger a server-side action for staff | Staff templates |

---

## 7. Example Decisions, Worked Out

These examples make the rules concrete.

### Example A — Password reset
- **Audience:** Any signed-in or signing-up user.
- **Mechanics:** Email link → form → token validation → password update → success page.
- **Decision:** Staff templates. Django ships every primitive for this. Rebuilding it in the cockpit adds risk (token handling, email flows) for no product gain.

### Example B — Customer support chat
- **Audience:** End customers.
- **Mechanics:** WebSocket to `ws/support_agent/<id>/`, streaming tokens, tool-call panel, intent/sentiment side panel.
- **Decision:** Cockpit. Real-time streaming and polished customer-facing UI are core cockpit territory.

### Example C — Staff dashboard with KPIs
- **Audience:** Internal operations team (≤ 10 users).
- **Mechanics:** A page of cards summarising today's conversations and top intents.
- **Decision:** Staff templates if it is purely internal and read-only. Cockpit if it is the same view external operators will use; in that case the staff template is replaced when the cockpit version ships.

### Example D — Bulk-edit a hundred products
- **Audience:** A single admin.
- **Decision:** Django admin. Custom UI is not justified.

### Example E — NLP playground (compare BERT vs GPT vs RAG on one input)
- **Audience:** NLP practitioners and curious operators.
- **Mechanics:** Streamed results, side-by-side panels, prompt iteration.
- **Decision:** Cockpit. The streaming and side-by-side comparison are precisely the kinds of interactions client rendering serves best.

### Example F — One-off "rebuild RAG index" trigger for staff
- **Audience:** Engineers and senior staff.
- **Mechanics:** A button that queues a Celery job and shows the run history.
- **Decision:** Staff templates. Quick to ship, no need for a cockpit route.

---

## 8. What This Costs (honestly)

It is fair to name the costs so contributors plan around them.

- **Two test stacks.** `pytest` in the backend, `vitest` in the cockpit. Both are already in the repo; the cost is steady-state, not setup.
- **Two visual languages.** Bootstrap 5 utilities for the staff tier; Tailwind tokens for the cockpit. The brand-level consistency rule (R7) keeps this from looking accidental.
- **Two deploy concerns.** Django runs on Daphne in the existing container. The cockpit is a Node build that ships separately. The cost is in the deploy pipeline, not in day-to-day development.
- **Reviewer vigilance.** R6 only works if reviewers actually ask "which tier and why?" on every PR.

These costs are real. They are the price of the benefits in §3 and we judge them worth paying.

---

## 9. What Would Make Us Reconsider

The architecture is not religious. We would revisit it if any of the following became true:

- **The staff tier has not been touched in six months** and every new internal need lands in the cockpit anyway. That would be evidence the cockpit has absorbed the staff role and the templates tier is dead weight.
- **The cockpit has implemented its own auth, password reset, and admin replacements.** That would be evidence the boundary moved and the Django tier should retire to admin-only.
- **An operator complains that switching tiers is jarring.** Brand-level consistency (R7) becomes pixel-level inconsistency in their hands.

If any of those happen, we open a written proposal and update this document. Until then, three tiers is the plan.

---

## 10. Bottom Line

The platform ships three tiers because real platforms have three audiences. Admin for power users, server-rendered staff workflows for internal users, and a polished cockpit for the product. Each tier uses the framework that fits its job, owns a fixed remit, and is bounded by an explicit contract. Done with discipline, this is **more** maintainable than a single hybrid frontend, not less — because every page has an obvious home and no two pages compete for it.
