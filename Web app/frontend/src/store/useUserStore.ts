import { create } from 'zustand';

export interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  token: string;
  reports: Report[];
  emailNotifications: boolean;
  darkMode: boolean; 
  meterData: {
    id: string;
    type: string;
    value: string | number;
    timestamp: string;
  }[];
}

export interface Report {
  id: string;
  title: string;
  date: string;
}

export interface UserStore {
  user: User | null;
  isLoading: boolean;
  error: string | null;
  isAuthenticated: boolean;

  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  fetchUser: () => Promise<void>;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setUser: (user: User | null) => void;
}

interface LoginResponse {
  user: User;
  token: string;
}

export const useUserStore = create<UserStore>((set) => ({
  user: null,
  isLoading: false,
  error: null,
  isAuthenticated: false,

  login: async (email: string, password: string): Promise<void> => {
    set({ isLoading: true });
    try {
      const response = await fetch('http://127.0.0.1:8000/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) throw new Error('Login failed');
      const data: LoginResponse = await response.json();
      
      set({
        user: data.user,
        isAuthenticated: true,
        isLoading: false,
      });
      
      localStorage.setItem('auth_token', data.token);
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false });
    }
  },

  logout: (): void => {
    set({ user: null, isAuthenticated: false });
    localStorage.removeItem('auth_token');
  },

  fetchUser: async (): Promise<void> => {
    set({ isLoading: true });
    try {
      const token = localStorage.getItem('auth_token');
      if (!token) throw new Error('No token found');

      const response = await fetch('http://127.0.0.1:8000/api/auth/me', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error('Failed to fetch user data');
      const data: { user: User } = await response.json();

      set({
        user: data.user,
        isAuthenticated: true,
        isLoading: false,
      });
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false });
    }
  },

  setLoading: (loading: boolean): void => set({ isLoading: loading }),

  setError: (error: string | null): void => set({ error }),
  setUser: (user: User | null): void => set({ user }),
}));
