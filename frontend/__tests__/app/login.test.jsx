import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock next/navigation
const mockPush = vi.fn();
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}));

// Mock the api module
vi.mock('@/lib/api', () => ({
  login: vi.fn(),
}));

import LoginPage from '@/app/login/page.jsx';
import { login } from '@/lib/api';

describe('LoginPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the login form with default values', () => {
    render(<LoginPage />);

    expect(screen.getByRole('heading', { name: /log in/i })).toBeInTheDocument();
    expect(screen.getByLabelText('Username')).toHaveValue('demo_user_01');
    expect(screen.getByLabelText('Password')).toHaveValue('demo12345');
    expect(screen.getByRole('button', { type: 'submit' })).toBeInTheDocument();
  });

  it('allows changing username and password', async () => {
    const user = userEvent.setup();
    render(<LoginPage />);

    const usernameInput = screen.getByLabelText('Username');
    const passwordInput = screen.getByLabelText('Password');

    await user.clear(usernameInput);
    await user.type(usernameInput, 'new_user');

    await user.clear(passwordInput);
    await user.type(passwordInput, 'new_pass');

    expect(usernameInput).toHaveValue('new_user');
    expect(passwordInput).toHaveValue('new_pass');
  });

  it('calls login and redirects on success', async () => {
    const user = userEvent.setup();
    login.mockResolvedValueOnce({ access: 'token', refresh: 'refresh' });
    render(<LoginPage />);

    await user.click(screen.getByRole('button', { type: 'submit' }));

    expect(login).toHaveBeenCalledWith('demo_user_01', 'demo12345');
    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith('/products');
    });
  });

  it('shows error message on login failure', async () => {
    const user = userEvent.setup();
    const error = new Error('API error 401');
    error.data = { detail: 'Invalid credentials' };
    login.mockRejectedValueOnce(error);
    render(<LoginPage />);

    await user.click(screen.getByRole('button', { type: 'submit' }));

    await waitFor(() => {
      expect(screen.getByText('Invalid credentials')).toBeInTheDocument();
    });
    expect(mockPush).not.toHaveBeenCalled();
  });

  it('shows generic error when no detail in error data', async () => {
    const user = userEvent.setup();
    const error = new Error('API error 500');
    error.data = null;
    login.mockRejectedValueOnce(error);
    render(<LoginPage />);

    await user.click(screen.getByRole('button', { type: 'submit' }));

    await waitFor(() => {
      expect(screen.getByText('Login failed')).toBeInTheDocument();
    });
  });

  it('disables button while logging in', async () => {
    const user = userEvent.setup();
    let resolveLogin;
    login.mockReturnValueOnce(new Promise((resolve) => { resolveLogin = resolve; }));
    render(<LoginPage />);

    const button = screen.getByRole('button', { type: 'submit' });
    await user.click(button);

    expect(screen.getByText('Logging in…')).toBeInTheDocument();
    expect(button).toBeDisabled();

    resolveLogin({ access: 't', refresh: 'r' });
    await waitFor(() => {
      expect(mockPush).toHaveBeenCalled();
    });
  });
});
