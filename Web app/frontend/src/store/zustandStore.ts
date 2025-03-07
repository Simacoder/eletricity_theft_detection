import { create } from 'zustand';

interface Store {
  user: { id: string; username: string } | null;
  setUser: (user: { id: string; username: string } | null) => void;
  meterData: { meter_id: string; value: number; timestamp: string }[];
  setMeterData: (data: { meter_id: string; value: number; timestamp: string }[]) => void;
  alerts: { id: string; message: string; severity: string }[];
  setAlerts: (alerts: { id: string; message: string; severity: string }[]) => void;
}

export const useStore = create<Store>((set) => ({
  user: null,
  setUser: (user) => set(() => ({ user })),
  meterData: [],
  setMeterData: (data) => set(() => ({ meterData: data })),
  alerts: [],
  setAlerts: (alerts) => set(() => ({ alerts }))
}));
