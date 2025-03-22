import { create } from 'zustand';

interface Alert {
  id: string;
  message: string;
  severity: string;
  timestamp: string;
  acknowledged: boolean;
}

interface AlertStore {
  alerts: Alert[];
  isLoading: boolean;
  error: string | null;

  fetchAlerts: () => Promise<void>;
  acknowledgeAlert: (alertId: string) => void;
  deleteAlert: (alertId: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useAlertStore = create<AlertStore>((set) => ({
  alerts: [],
  isLoading: false,
  error: null,

  fetchAlerts: async () => {
    set({ isLoading: true });
    try {
      const response = await fetch('http://127.0.0.1:8000/api/alerts');
      if (!response.ok) throw new Error('Failed to fetch alerts');
      const data = await response.json();
      set({ alerts: data, isLoading: false });
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false });
    }
  },

  acknowledgeAlert: (alertId: string) => {
    set((state) => {
      const updatedAlerts = state.alerts.map((alert) =>
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      );
      return { alerts: updatedAlerts };
    });
  },

  deleteAlert: (alertId: string) => {
    set((state) => ({
      alerts: state.alerts.filter((alert) => alert.id !== alertId),
    }));
  },

  setLoading: (loading: boolean) => set({ isLoading: loading }),

  setError: (error: string | null) => set({ error }),
}));
