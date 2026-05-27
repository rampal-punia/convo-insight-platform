# ConvoInsight Frontend (Next.js, JSX)

Pure JavaScript Next.js 15 (App Router) skeleton — **no TypeScript**.

## Quick start

```bash
cd frontend
cp .env.example .env.local        # point at your backend (default localhost:8000)
npm install
npm run dev                       # http://localhost:3000
```

The backend must be running (`make up` or `python manage.py runserver`) and you should
seed demo data first:

```bash
# From the repo root, with your venv active:
python manage.py seed_demo
```

Then log in at `http://localhost:3000/login` with username `demo_user_01` and password
`demo12345`.

## What's inside

| Path                          | Purpose                                                 |
| ----------------------------- | ------------------------------------------------------- |
| `src/lib/api.js`              | Fetch wrapper with JWT attach + auto-refresh on 401     |
| `src/app/layout.jsx`          | Root layout (Tailwind, global CSS)                      |
| `src/app/page.jsx`            | Landing page                                            |
| `src/app/login/page.jsx`      | JWT login form                                          |
| `src/app/products/page.jsx`   | Authenticated list of `/api/v1/products/`               |

## Conventions

- **JSX only** — never create `.ts` or `.tsx` files. Use JSDoc for typing hints if needed.
- Path alias `@/*` resolves to `src/*` (configured in `jsconfig.json`).
- Auth tokens live in `localStorage` (intentionally simple for dev — production should
  migrate to httpOnly cookies set by Django).

## Where to look in the backend

- API base: `http://localhost:8000/api/v1/`
- Swagger UI: `http://localhost:8000/api/docs/`
- OpenAPI schema: `http://localhost:8000/api/schema/`
- JWT endpoints: `/api/v1/auth/token/`, `/api/v1/auth/token/refresh/`, `/api/v1/auth/token/blacklist/`
