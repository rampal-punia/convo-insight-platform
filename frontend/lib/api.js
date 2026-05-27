/**
 * Thin fetch wrapper around the Django REST API.
 *
 * Handles:
 *   - JWT access token attachment
 *   - Automatic refresh on 401
 *   - JSON serialization
 *
 * Tokens are kept in localStorage for simplicity. For production, prefer
 * httpOnly cookies set by the backend.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const ACCESS_KEY = 'ci_access';
const REFRESH_KEY = 'ci_refresh';

export function getAccess() {
  if (typeof window === 'undefined') return null;
  return window.localStorage.getItem(ACCESS_KEY);
}

export function getRefresh() {
  if (typeof window === 'undefined') return null;
  return window.localStorage.getItem(REFRESH_KEY);
}

export function setTokens({ access, refresh }) {
  if (typeof window === 'undefined') return;
  if (access) window.localStorage.setItem(ACCESS_KEY, access);
  if (refresh) window.localStorage.setItem(REFRESH_KEY, refresh);
}

export function clearTokens() {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(ACCESS_KEY);
  window.localStorage.removeItem(REFRESH_KEY);
}

async function refreshAccessToken() {
  const refresh = getRefresh();
  if (!refresh) return null;
  const resp = await fetch(`${API_URL}/api/v1/auth/token/refresh/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh }),
  });
  if (!resp.ok) {
    clearTokens();
    return null;
  }
  const data = await resp.json();
  setTokens(data);
  return data.access;
}

/**
 * Generic API call. Path is relative (e.g. "/api/v1/products/").
 */
export async function api(path, { method = 'GET', body, headers = {}, auth = true } = {}) {
  const url = path.startsWith('http') ? path : `${API_URL}${path}`;
  const opts = {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
  };
  if (auth) {
    const access = getAccess();
    if (access) opts.headers.Authorization = `Bearer ${access}`;
  }
  if (body !== undefined) opts.body = JSON.stringify(body);

  let resp = await fetch(url, opts);

  if (resp.status === 401 && auth) {
    const newAccess = await refreshAccessToken();
    if (newAccess) {
      opts.headers.Authorization = `Bearer ${newAccess}`;
      resp = await fetch(url, opts);
    }
  }

  const contentType = resp.headers.get('content-type') || '';
  const payload = contentType.includes('application/json') ? await resp.json() : await resp.text();

  if (!resp.ok) {
    const error = new Error(`API error ${resp.status}`);
    error.status = resp.status;
    error.data = payload;
    throw error;
  }
  return payload;
}

export async function login(username, password) {
  const data = await api('/api/v1/auth/token/', {
    method: 'POST',
    body: { username, password },
    auth: false,
  });
  setTokens(data);
  return data;
}

export async function logout() {
  const refresh = getRefresh();
  if (refresh) {
    try {
      await api('/api/v1/auth/token/blacklist/', { method: 'POST', body: { refresh } });
    } catch {
      /* ignore */
    }
  }
  clearTokens();
}

export async function me() {
  return api('/api/v1/users/me/');
}
