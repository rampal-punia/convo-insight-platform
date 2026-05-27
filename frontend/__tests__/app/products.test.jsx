import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock next/navigation
const mockPush = vi.fn();
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}));

// Mock the api module — api() must always return a promise to avoid
// "Cannot read properties of undefined (reading 'then')" when the
// component calls api(...).then(...) in useEffect.
const mockApi = vi.fn().mockResolvedValue({ results: [] });
const mockGetAccess = vi.fn();
const mockLogout = vi.fn();

vi.mock('@/lib/api', () => ({
  api: (...args) => mockApi(...args),
  getAccess: () => mockGetAccess(),
  logout: () => mockLogout(),
}));

import ProductsPage from '@/app/products/page.jsx';

const SAMPLE_PRODUCTS = {
  results: [
    { id: 1, name: 'MacBook Pro', description: 'Laptop', price: '1999.00', stock: 50 },
    { id: 2, name: 'iPhone 16', description: 'Phone', price: '999.00', stock: 100 },
  ],
};

describe('ProductsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset default mock implementation
    mockApi.mockResolvedValue({ results: [] });
  });

  it('redirects to /login if no access token', () => {
    mockGetAccess.mockReturnValueOnce(null);
    render(<ProductsPage />);

    expect(mockPush).toHaveBeenCalledWith('/login');
  });

  it('shows loading state then renders products', async () => {
    mockGetAccess.mockReturnValue('valid-token');
    mockApi.mockResolvedValueOnce(SAMPLE_PRODUCTS);
    render(<ProductsPage />);

    expect(screen.getByText('Loading…')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('MacBook Pro')).toBeInTheDocument();
      expect(screen.getByText('iPhone 16')).toBeInTheDocument();
      expect(screen.getByText('$1999.00')).toBeInTheDocument();
      expect(screen.getByText('stock: 50')).toBeInTheDocument();
    });
  });

  it('shows error message when API fails', async () => {
    mockGetAccess.mockReturnValue('valid-token');
    const error = new Error('API error 500');
    error.data = { detail: 'Server error' };
    mockApi.mockRejectedValueOnce(error);
    render(<ProductsPage />);

    await waitFor(() => {
      expect(screen.getByText('Server error')).toBeInTheDocument();
    });
  });

  it('shows fallback error when no detail in error', async () => {
    mockGetAccess.mockReturnValue('valid-token');
    const error = new Error('Network failed');
    error.data = null;
    mockApi.mockRejectedValueOnce(error);
    render(<ProductsPage />);

    await waitFor(() => {
      expect(screen.getByText('Network failed')).toBeInTheDocument();
    });
  });

  it('shows empty state when no products exist', async () => {
    mockGetAccess.mockReturnValue('valid-token');
    mockApi.mockResolvedValueOnce({ results: [] });
    render(<ProductsPage />);

    await waitFor(() => {
      expect(screen.getByText(/no products yet/i)).toBeInTheDocument();
    });
  });

  it('calls logout and redirects when logout button is clicked', async () => {
    const user = userEvent.setup();
    mockGetAccess.mockReturnValue('valid-token');
    mockApi.mockResolvedValueOnce(SAMPLE_PRODUCTS);
    render(<ProductsPage />);

    await waitFor(() => {
      expect(screen.getByText('MacBook Pro')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Log out'));

    expect(mockLogout).toHaveBeenCalled();
    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith('/login');
    });
  });

  it('renders correct number of product items', async () => {
    mockGetAccess.mockReturnValue('valid-token');
    mockApi.mockResolvedValueOnce(SAMPLE_PRODUCTS);
    render(<ProductsPage />);

    await waitFor(() => {
      const items = screen.getAllByRole('listitem');
      expect(items).toHaveLength(2);
    });
  });
});
