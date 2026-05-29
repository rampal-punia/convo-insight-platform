# Auth.js Integration — Implementation Playbook

> A step-by-step doc for interns to implement **Auth.js v5 (`next-auth@5.0.0-beta.31`)** as the cockpit's session layer, against the existing Django JWT backend. No code is included on purpose: this is the **plan**. You write the code.
>
> When you finish, the cockpit will no longer keep tokens in `localStorage`. Sessions will live in an encrypted, httpOnly cookie managed by Auth.js, refresh-token rotation will be handled in the `jwt` callback, route protection will be middleware-based, and WebSocket connections will continue to receive a valid JWT via `?token=`.

---

## 0. Read First

Before you start, read these in order:

1. [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) — three tiers, contract surface, auth posture.
2. [frontend/docs/VISION.md](./VISION.md) and [frontend/docs/STRATEGY.md](./STRATEGY.md) — why this work matters, where it sits in the phases.
3. [frontend/docs/CURRENT_STATUS.md](./CURRENT_STATUS.md) — what exists today.
4. The Auth.js docs (v5 beta): https://authjs.dev — focus on **Getting Started → Installation**, **Authentication → Credentials**, **Session strategies → JWT**, and **Reference → Callbacks**.

---

## 1. Why Auth.js (and why not just keep `localStorage`)

Today the cockpit stores the Django-issued JWT pair in `localStorage`. That works in development but has three real problems:

- **XSS exposure.** Any script that runs on the page can read `localStorage`.
- **Per-page auth guards.** Every protected route re-implements `if (!getAccess()) router.push('/login')`.
- **Refresh logic lives in `fetch` retries.** There is no clean lifecycle hook for "the session expired" beyond a 401 on the next request.

Auth.js v5 fixes all three:

- Sessions live in an **encrypted, httpOnly cookie** (`__Secure-authjs.session-token` in production).
- A single **`middleware.js`** at the project root protects routes by matching paths.
- The **`jwt` callback** runs on every session read, which is the canonical place to refresh expired access tokens silently.
- The same pattern extends to OAuth providers (Google, GitHub) later — one line per provider.

Django remains the **canonical identity store**. Auth.js never owns identity; it owns the cockpit's session.

---

## 2. The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Cockpit (Next.js 16)                   │
│                                                                 │
│   pages / components ─────► auth() / useSession()               │
│           │                                                     │
│           ▼                                                     │
│   middleware.js (route protection)                              │
│           │                                                     │
│           ▼                                                     │
│   Auth.js handler at app/api/auth/[...nextauth]/route.js        │
│      • Credentials provider                                     │
│      • jwt callback   ──► refresh against Django                │
│      • session callback ─► expose access token to client/server │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  │  encrypted session cookie
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Django backend                         │
│  POST /api/v1/auth/token/           → {access, refresh}         │
│  POST /api/v1/auth/token/refresh/   → {access}                  │
│  POST /api/v1/auth/token/blacklist/ → 200                       │
│  GET  /api/v1/users/me/             → current user              │
│  GET  /api/v1/*                     → resource APIs             │
│  WS   ws/*  ?token=<access>         → consumers                 │
└─────────────────────────────────────────────────────────────────┘
```

The Django backend is **unchanged**. Auth.js calls the same JWT endpoints the cockpit calls today.

---

## 3. Prerequisites

- `next-auth@5.0.0-beta.31` is already in `package.json`. After `npm install`, you have it.
- Generate a session secret. Use one of:
  - `openssl rand -base64 32`
  - `npx auth secret` (writes a `.env.local` automatically)
- Backend running on `http://localhost:8000` with `seed_demo`, `create_intents`, `create_topics`, `populate_rag_store` already executed.

Environment variables you will add to `frontend/.env.local`:

| Variable | Purpose |
|---|---|
| `AUTH_SECRET` | 32-byte secret used to encrypt the session cookie. Mandatory in production; recommended in development. |
| `AUTH_TRUST_HOST` | Set to `true` in non-Vercel deployments behind a proxy. Optional locally. |
| `NEXT_PUBLIC_API_URL` | Django base URL (already in use). |
| `INTERNAL_API_URL` | Server-side Django base URL. Same as `NEXT_PUBLIC_API_URL` in development; differs in Docker / production where the cockpit and backend are on different networks. |

Document any new env var in `frontend/.env.example` in the same PR.

---

## 4. Files You Will Create

This is the full file layout you are targeting. Nothing more, nothing less.

```
frontend/
├── auth.js                                 # createAuth() config + handlers
├── middleware.js                           # route protection
├── app/
│   ├── api/
│   │   └── auth/
│   │       └── [...nextauth]/
│   │           └── route.js                # re-exports handlers from auth.js
│   ├── login/
│   │   └── page.jsx                        # rewritten to call signIn('credentials', …)
│   └── (app)/                              # (future) route group whose layout uses auth()
│       └── layout.jsx                      # server component that calls auth()
├── lib/
│   ├── auth-server.js                      # thin server-side helpers wrapping auth()
│   ├── api.js                              # rewritten: no localStorage, reads token from session
│   └── ws.js                               # (future, Phase 2) reads token from session for WS
└── __tests__/
    ├── auth.test.js                        # callback logic tests (jwt + session)
    └── middleware.test.js                  # smoke test for protected paths
```

> The `app/(app)/` route group is the **target layout** from [STRATEGY.md](./STRATEGY.md) §3. You will introduce it in the same PR that introduces Auth.js, because the route group is where the server-side auth guard naturally lives.

---

## 5. Step-by-Step Implementation Plan

Work in this order. Do not skip ahead. Each step ends with a verifiable outcome.

### Step 1 — Bootstrap

1. Run `npm install` in `frontend/` (next-auth is already in `package.json`).
2. Create `frontend/.env.local` with `AUTH_SECRET`, `NEXT_PUBLIC_API_URL`, and `INTERNAL_API_URL`.
3. Add the same keys to `frontend/.env.example` with placeholder values.

**Verify:** `npm run build` still passes.

### Step 2 — Create `auth.js` at the project root

This is the **single configuration file** for Auth.js v5. Inside it you will:

1. Import `NextAuth` from `next-auth` and `Credentials` from `next-auth/providers/credentials`.
2. Call `NextAuth({...})` and export `{ handlers, auth, signIn, signOut }`.
3. Configure:
   - `session.strategy = 'jwt'` — required because we are not using a database adapter.
   - `pages.signIn = '/login'` — sends unauthenticated users to your custom login page.
   - `providers = [Credentials({ ... })]`.
4. Implement the Credentials provider's `authorize(credentials)`:
   - Expect `username` and `password` in `credentials`.
   - `POST` them to `INTERNAL_API_URL + '/api/v1/auth/token/'`.
   - If the response is OK, fetch `/api/v1/users/me/` with the new access token to get the user object.
   - Return an object with `{ id, name, email, accessToken, refreshToken, accessTokenExpires }` where `accessTokenExpires` is `Date.now() + access_token_lifetime_seconds * 1000` (decode the JWT `exp` claim or use the documented 5-minute lifetime).
   - Return `null` on any failure. Do not throw.
5. Implement the `jwt` callback. This runs on every request that needs the session.
   - On **sign-in** (`user` is present), copy `accessToken`, `refreshToken`, `accessTokenExpires`, and user info into the `token`.
   - On **subsequent calls**, if `Date.now() < token.accessTokenExpires - 60_000` (60 s safety margin), return the token unchanged.
   - Otherwise, `POST` to `/api/v1/auth/token/refresh/` with `token.refreshToken`. If successful, update `token.accessToken` and `token.accessTokenExpires`; rotate `token.refreshToken` only if the backend returns a new one.
   - If the refresh fails, set `token.error = 'RefreshAccessTokenError'` and return the token (do not throw).
6. Implement the `session` callback.
   - Copy `accessToken`, the user object, and `token.error` onto `session`.
   - Do **not** expose `refreshToken` to the client. The session is read on the client; the refresh token must stay server-side.
7. Set `trustHost: true` if `process.env.AUTH_TRUST_HOST === 'true'`.

**Verify:** `npm run build` passes; no runtime check yet.

### Step 3 — Wire the Auth.js handler

Create `app/api/auth/[...nextauth]/route.js`. It only re-exports the `handlers` (`GET` and `POST`) imported from your `auth.js`. That is the only line of work in this file.

**Verify:** Hit `http://localhost:3000/api/auth/providers` in the browser; you should see a JSON document listing the Credentials provider.

### Step 4 — Add `middleware.js`

At the project root, create `middleware.js` that:

1. Imports `auth` from `./auth.js` (or imports a tiny `authConfig` slice — see Auth.js's "Edge compatibility" docs).
2. Exports `default auth` as the middleware function.
3. Exports a `config.matcher` array that protects everything except: `/`, `/login`, `/api/auth/*`, Next.js internals (`/_next/*`), static files, and the favicon.

The matcher pattern is the canonical Auth.js v5 example; copy it from the docs and adapt the public path list.

**Verify:** Visit `http://localhost:3000/products` without signing in. You should be redirected to `/login?callbackUrl=…`.

### Step 5 — Rewrite the login page

Open `app/login/page.jsx`. Replace the current call to `login(username, password)` with a call to `signIn('credentials', { username, password, redirect: false, callbackUrl: '/products' })` imported from `next-auth/react`.

- On success (`result.ok`), navigate to `result.url ?? '/products'`.
- On failure (`result?.error`), show the existing error UI.
- Wrap the page (or the root layout) with `SessionProvider` from `next-auth/react` so client components can use `useSession()`.

Remove all references to `getAccess()`, `setTokens()`, and `clearTokens()` from this page.

**Verify:** Sign in with `demo_user_01` / `demo12345`. You should land on `/products`. The browser **must not** have anything in `localStorage` under `ci_access` or `ci_refresh`. There **must** be an `authjs.session-token` cookie.

### Step 6 — Rewrite `lib/api.js` to be session-aware

The new `api(path, options)` function does **not** read tokens from `localStorage`. Instead:

- **In server components and route handlers**, accept an optional `accessToken` argument supplied by the caller (which got it from `auth()`).
- **In client components**, fall back to calling `/api/auth/session` (Auth.js exposes the session at this URL) to obtain the current `accessToken`, then attach it as `Authorization: Bearer <token>`.
- Remove the `refreshAccessToken()` function entirely — refresh now happens inside the `jwt` callback.
- On a `401` response, call `signIn()` or set `window.location.href = '/login'` and stop. Do not retry.

Update every page that imports from `lib/api.js` accordingly:

- `app/products/page.jsx` — fetch with the session-aware helper.
- Future authenticated pages — same pattern.

**Verify:** `/products` still lists products after sign-in. Network tab shows the `Authorization` header on every request to `/api/v1/*`.

### Step 7 — Replace per-page guards with the route-group layout

1. Create `app/(app)/layout.jsx` as a **server component**.
2. Inside, call `await auth()`. If it returns `null`, `redirect('/login')` (from `next/navigation`).
3. Move `app/products/page.jsx` to `app/(app)/products/page.jsx`.
4. Remove the inline `getAccess()` check from the products page; it is now redundant.
5. The middleware handles unauthenticated access at the edge; the layout handles the rare case where middleware did not catch it (e.g. local rewrites).

**Verify:** Logged-in users still see `/products`. Logged-out users are redirected at the middleware level **and** the server-side `auth()` check.

### Step 8 — Implement sign-out properly

In whatever component exposes a "Log out" control:

1. Call `signOut({ redirect: false })` from `next-auth/react` to clear the cookie.
2. After it resolves, `POST` to `/api/v1/auth/token/blacklist/` with the previous `refreshToken` if you can still access it server-side (you cannot, because the session is gone). The cleanest approach is to expose a small `POST /api/auth/signout` route that calls the blacklist endpoint **before** invoking Auth.js's sign-out; or run the blacklist in the `signOut` event hook of `auth.js` (see the Auth.js v5 events docs).
3. `router.push('/login')`.

**Verify:** After clicking "Log out", the session cookie is gone and the Django backend has blacklisted the refresh token (check the Django admin → Token Blacklist).

### Step 9 — Surface session in WebSockets (preparation for Phase 2)

You will not build the WebSocket client in this PR — `lib/ws.js` is a Phase 2 deliverable. But document the pattern in this file so the next PR can implement it:

- Server components that need a WebSocket URL call `auth()` and pass `session.accessToken` into the URL.
- Client components that need a WebSocket URL call `useSession()`, wait for `status === 'authenticated'`, then read `data.accessToken`.

That keeps the cockpit's single source of session truth — the Auth.js cookie — even on the real-time path.

### Step 10 — Tests

Add at least the following Vitest tests:

- `__tests__/auth.test.js` — mock `fetch`; verify the `authorize` callback returns the expected shape on success and `null` on failure; verify the `jwt` callback refreshes when expired and surfaces `RefreshAccessTokenError` on refresh failure.
- `__tests__/middleware.test.js` — smoke test confirming `middleware.js` exports a default function and a `config.matcher` of the expected shape.
- Update `__tests__/app/login.test.jsx` to mock `next-auth/react`'s `signIn` instead of `lib/api.js`'s `login`.
- Update `__tests__/app/products.test.jsx` to mock the session-aware `api()` helper.

**Verify:** `npm test` passes with the same or higher coverage as before.

### Step 11 — Documentation sweep

In the same PR, update:

- `frontend/docs/FRONTEND_GUIDE.md` — auth flow section, code patterns.
- `frontend/docs/CURRENT_STATUS.md` — mark the migration as complete; remove "JWT tokens in `localStorage`" from the active phase checklist.
- `frontend/docs/CONTRIBUTING.md` — non-negotiable #2 now reads "Use the shared session via `auth()` / `useSession()`; never call `signIn`/`signOut` directly from a non-auth page".
- `frontend/.env.example` — `AUTH_SECRET`, `INTERNAL_API_URL`, optionally `AUTH_TRUST_HOST`.

**Verify:** No mention of `localStorage`, `getAccess`, `ci_access`, or `ci_refresh` remains anywhere under `frontend/`.

---

## 6. Architectural Rules This Migration Locks In

Once Auth.js is in place, these become non-negotiable in review:

1. **No direct token storage.** The cockpit never writes to `localStorage` or `sessionStorage` for auth.
2. **One session, one helper.** Server code uses `auth()`; client code uses `useSession()`. Nothing else.
3. **Route protection is middleware-first.** The `(app)/layout.jsx` server check is a belt-and-braces second line.
4. **Refresh is invisible.** The `jwt` callback owns it. No `fetch` call retries with a refreshed token at the call site.
5. **The Django backend stays the canonical identity store.** Auth.js does not replicate users or maintain its own user table.
6. **OAuth, if added, is a one-provider PR.** Add `Google`/`GitHub` to `providers`, update the Django backend to accept the OAuth-issued identity, document the trade-off in the decision log. Do not bolt OAuth on inside this PR.

---

## 7. Pitfalls to Avoid

The Auth.js v5 beta has sharp edges. These are the ones interns hit most often.

- **Middleware on the edge runtime.** If your `auth.js` imports anything Node-only (a CSS-in-JS helper, a DB driver, `crypto`-using code), middleware will fail to compile. Keep `auth.js` minimal and split out an `authConfig.js` slice if needed — the Auth.js docs show this split.
- **`AUTH_SECRET` missing in production.** The cookie cannot be encrypted without it; sign-in silently fails.
- **`trustHost` behind a proxy.** Without `AUTH_TRUST_HOST=true`, Auth.js refuses to issue cookies under reverse proxies that rewrite `Host`.
- **Refresh-token rotation.** Django simplejwt can be configured to rotate refresh tokens. If it does, you must update `token.refreshToken` in the `jwt` callback or the next refresh will fail.
- **Server fetch URL.** In Docker the cockpit cannot reach `http://localhost:8000`; that is why `INTERNAL_API_URL` exists. Use it on the server, `NEXT_PUBLIC_API_URL` on the client.
- **Cookie size.** Putting large user objects in the session inflates the cookie. Keep `session.user` to id, name, email, and the access token only.
- **Logging the token.** The session payload contains the access token. Never log the session object verbatim.
- **WebSocket reconnect after refresh.** When the access token rotates, existing WS connections still hold the old token. The Phase 2 `lib/ws.js` must close and reconnect on session change.
- **Test isolation.** Reset `fetch` mocks between tests; the `jwt` callback caches the token across calls in module scope if you write it that way.

---

## 8. How You'll Know You're Done

Tick this list before opening the PR.

- [ ] `npm install && npm run build && npm test` are all green.
- [ ] No `localStorage` / `sessionStorage` access in the cockpit. (`grep -RIn 'localStorage' frontend/app frontend/lib` returns nothing.)
- [ ] No usage of the old `login`, `logout`, `getAccess`, `setTokens`, `clearTokens` exports. (`grep` confirms.)
- [ ] `middleware.js` protects `/products` and any other authenticated route; visiting them logged out redirects to `/login?callbackUrl=…`.
- [ ] Signing in with `demo_user_01` / `demo12345` lands on `/products` and the only persistent auth artefact in DevTools is the `authjs.session-token` cookie (or `__Secure-authjs.session-token` in production).
- [ ] After 5 minutes idle, a fresh request to `/api/v1/products/` still succeeds — the `jwt` callback refreshed the access token silently.
- [ ] Signing out clears the cookie and blacklists the refresh token on Django.
- [ ] All four docs in §5 step 11 are updated in the same PR.

---

## 9. After This PR

Two follow-ups are queued, not in scope here:

1. **`lib/ws.js`** in Phase 2 reads `data.accessToken` from the session and attaches it via `?token=` to every WebSocket URL. The `useSession()` hook tells the WS client when to reconnect after a refresh.
2. **OAuth providers** when product needs them. The pattern is one extra entry in the `providers` array plus a small Django change to accept the OAuth identity. Document the decision in `frontend/docs/decisions.md`.

---

## 10. Reference Reading

- Auth.js v5 — https://authjs.dev (Installation, Credentials provider, JWT session, Callbacks, Middleware, Events)
- Auth.js v5 migration guide — https://authjs.dev/getting-started/migrating-to-v5
- Next.js 16 Middleware — https://nextjs.org/docs/app/building-your-application/routing/middleware
- Django simplejwt — token lifetime, rotation, blacklist: https://django-rest-framework-simplejwt.readthedocs.io/

That is the whole plan. When you have shipped it, the cockpit is on the same identity story we will keep into production.
