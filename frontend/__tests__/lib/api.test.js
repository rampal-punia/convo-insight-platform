import { describe, it, expect, beforeEach, vi } from 'vitest';

// Must import before the module under test so the mock is in place
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

// Helpers to build mock Response objects
function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

// Re-import the module for each test so module-level state is fresh
const {
  getAccess,
  getRefresh,
  setTokens,
  clearTokens,
  api,
  login,
  logout,
  me,
} = await import('@/lib/api');

describe('lib/api — token helpers', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('setTokens writes to localStorage', () => {
    setTokens({ access: 'acc123', refresh: 'ref456' });
    expect(localStorage.getItem('ci_access')).toBe('acc123');
    expect(localStorage.getItem('ci_refresh')).toBe('ref456');
  });

  it('getAccess returns null when nothing is stored', () => {
    expect(getAccess()).toBeNull();
  });

  it('getAccess returns the stored token', () => {
    localStorage.setItem('ci_access', 'my-token');
    expect(getAccess()).toBe('my-token');
  });

  it('getRefresh returns the stored refresh token', () => {
    localStorage.setItem('ci_refresh', 'my-refresh');
    expect(getRefresh()).toBe('my-refresh');
  });

  it('clearTokens removes both tokens', () => {
    setTokens({ access: 'a', refresh: 'r' });
    clearTokens();
    expect(localStorage.getItem('ci_access')).toBeNull();
    expect(localStorage.getItem('ci_refresh')).toBeNull();
  });
});

describe('lib/api — api()', () => {
  beforeEach(() => {
    localStorage.clear();
    mockFetch.mockReset();
  });

  it('makes an authenticated GET request', async () => {
    localStorage.setItem('ci_access', 'test-access');
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ results: ['product1'] })
    );

    const data = await api('/api/v1/products/');
    expect(data).toEqual({ results: ['product1'] });

    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toBe('http://localhost:8000/api/v1/products/');
    expect(opts.method).toBe('GET');
    expect(opts.headers.Authorization).toBe('Bearer test-access');
  });

  it('does not attach Authorization when auth=false', async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true }));

    await api('/api/v1/auth/token/', { method: 'POST', auth: false });

    const [, opts] = mockFetch.mock.calls[0];
    expect(opts.headers.Authorization).toBeUndefined();
  });

  it('sends JSON body when provided', async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true }));

    await api('/test/', { method: 'POST', body: { name: 'foo' } });

    const [, opts] = mockFetch.mock.calls[0];
    expect(opts.body).toBe(JSON.stringify({ name: 'foo' }));
  });

  it('uses full URL when path starts with http', async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true }));

    await api('https://other.api.com/data');

    const [url] = mockFetch.mock.calls[0];
    expect(url).toBe('https://other.api.com/data');
  });

  it('throws with status and data on non-2xx response', async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ detail: 'Not found' }, 404)
    );

    try {
      await api('/api/v1/products/999/');
      expect.fail('should have thrown');
    } catch (err) {
      expect(err.status).toBe(404);
      expect(err.data).toEqual({ detail: 'Not found' });
      expect(err.message).toBe('API error 404');
    }
  });

  it('refreshes access token on 401 and retries', async () => {
    localStorage.setItem('ci_access', 'expired');
    localStorage.setItem('ci_refresh', 'valid-refresh');

    // First call returns 401
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ detail: 'Token expired' }, 401)
    );
    // Refresh call succeeds
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ access: 'new-access', refresh: 'new-refresh' })
    );
    // Retry with new token succeeds
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ results: ['ok'] })
    );

    const data = await api('/api/v1/products/');
    expect(data).toEqual({ results: ['ok'] });

    // Should have called fetch 3 times: original, refresh, retry
    expect(mockFetch).toHaveBeenCalledTimes(3);

    // Verify the new token was stored
    expect(localStorage.getItem('ci_access')).toBe('new-access');
  });

  it('clears tokens and throws if refresh also fails', async () => {
    localStorage.setItem('ci_access', 'expired');
    localStorage.setItem('ci_refresh', 'bad-refresh');

    // First call returns 401
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ detail: 'Token expired' }, 401)
    );
    // Refresh call fails
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ detail: 'Invalid refresh' }, 401)
    );

    try {
      await api('/api/v1/products/');
      expect.fail('should have thrown');
    } catch (err) {
      expect(err.status).toBe(401);
    }

    // Tokens should be cleared
    expect(localStorage.getItem('ci_access')).toBeNull();
    expect(localStorage.getItem('ci_refresh')).toBeNull();
  });
});

describe('lib/api — login()', () => {
  beforeEach(() => {
    localStorage.clear();
    mockFetch.mockReset();
  });

  it('posts credentials and stores tokens', async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ access: 'new-acc', refresh: 'new-ref' })
    );

    const data = await login('demo_user', 'pass123');

    expect(data).toEqual({ access: 'new-acc', refresh: 'new-ref' });
    expect(localStorage.getItem('ci_access')).toBe('new-acc');
    expect(localStorage.getItem('ci_refresh')).toBe('new-ref');

    // Verify it called the right endpoint without auth header
    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toContain('/api/v1/auth/token/');
    expect(opts.headers.Authorization).toBeUndefined();
    expect(JSON.parse(opts.body)).toEqual({
      username: 'demo_user',
      password: 'pass123',
    });
  });
});

describe('lib/api — logout()', () => {
  beforeEach(() => {
    localStorage.clear();
    mockFetch.mockReset();
  });

  it('blacklists the refresh token and clears storage', async () => {
    localStorage.setItem('ci_access', 'acc');
    localStorage.setItem('ci_refresh', 'ref');

    mockFetch.mockResolvedValueOnce(jsonResponse({}));

    await logout();

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, opts] = mockFetch.mock.calls[0];
    expect(url).toContain('/api/v1/auth/token/blacklist/');
    expect(JSON.parse(opts.body)).toEqual({ refresh: 'ref' });

    expect(localStorage.getItem('ci_access')).toBeNull();
    expect(localStorage.getItem('ci_refresh')).toBeNull();
  });

  it('still clears tokens even if blacklist call fails', async () => {
    localStorage.setItem('ci_access', 'acc');
    localStorage.setItem('ci_refresh', 'ref');

    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    await logout();

    expect(localStorage.getItem('ci_access')).toBeNull();
    expect(localStorage.getItem('ci_refresh')).toBeNull();
  });
});

describe('lib/api — me()', () => {
  beforeEach(() => {
    localStorage.clear();
    mockFetch.mockReset();
  });

  it('calls /api/v1/users/me/', async () => {
    localStorage.setItem('ci_access', 'token');
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ id: 1, username: 'alice' })
    );

    const data = await me();
    expect(data).toEqual({ id: 1, username: 'alice' });
    expect(mockFetch.mock.calls[0][0]).toContain('/api/v1/users/me/');
  });
});
