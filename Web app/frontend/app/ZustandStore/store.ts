import { create } from 'zustand';

type Store = {
  isLoginModalOpen: boolean;
  isRegistrationModalOpen: boolean;
  openLoginModal: () => void;
  openRegistrationModal: () => void;
  closeModals: () => void;
};

export const useStore = create<Store>((set) => ({
  isLoginModalOpen: false,
  isRegistrationModalOpen: false,
  openLoginModal: () => set({ isLoginModalOpen: true, isRegistrationModalOpen: false }),
  openRegistrationModal: () => set({ isLoginModalOpen: false, isRegistrationModalOpen: true }),
  closeModals: () => set({ isLoginModalOpen: false, isRegistrationModalOpen: false }),
}));
