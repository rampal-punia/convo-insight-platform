import Link from 'next/link';

export default function HomePage() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16">
      <h1 className="text-4xl font-bold tracking-tight">ConvoInsight</h1>
      <p className="mt-3 text-lg text-slate-600">
        Frontend skeleton for the conversational intelligence platform.
      </p>

      <div className="mt-10 grid gap-4 sm:grid-cols-2">
        <Link
          href="/login"
          className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm hover:border-slate-300"
        >
          <h2 className="text-xl font-semibold">Log in</h2>
          <p className="mt-1 text-sm text-slate-600">Get a JWT and start exploring the API.</p>
        </Link>
        <Link
          href="/products"
          className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm hover:border-slate-300"
        >
          <h2 className="text-xl font-semibold">Products demo</h2>
          <p className="mt-1 text-sm text-slate-600">Authenticated read from /api/v1/products/.</p>
        </Link>
      </div>

      <p className="mt-10 text-sm text-slate-500">
        Backend Swagger UI:{' '}
        <a
          href={`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/docs/`}
          target="_blank"
          rel="noreferrer"
          className="text-blue-600 underline"
        >
          /api/docs/
        </a>
      </p>
    </main>
  );
}
