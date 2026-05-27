'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { api, getAccess, logout } from '@/lib/api';

export default function ProductsPage() {
  const router = useRouter();
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!getAccess()) {
      router.push('/login');
      return;
    }
    api('/api/v1/products/')
      .then(setData)
      .catch((err) => setError(err.data?.detail || err.message));
  }, [router]);

  async function onLogout() {
    await logout();
    router.push('/login');
  }

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Products</h1>
        <button onClick={onLogout} className="text-sm text-slate-600 underline">
          Log out
        </button>
      </header>

      {error && <p className="mt-4 rounded bg-red-50 px-3 py-2 text-red-700">{error}</p>}
      {!data && !error && <p className="mt-4 text-slate-500">Loading…</p>}

      {data && (
        <ul className="mt-6 divide-y divide-slate-200 rounded-lg border border-slate-200 bg-white">
          {data.results.length === 0 && (
            <li className="px-4 py-6 text-slate-500">
              No products yet. Run <code className="rounded bg-slate-100 px-1">python manage.py seed_demo</code>.
            </li>
          )}
          {data.results.map((p) => (
            <li key={p.id} className="flex items-center justify-between px-4 py-3">
              <div>
                <p className="font-medium">{p.name}</p>
                <p className="text-sm text-slate-500">{p.description}</p>
              </div>
              <div className="text-right">
                <p className="font-semibold">${p.price}</p>
                <p className="text-xs text-slate-500">stock: {p.stock}</p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
