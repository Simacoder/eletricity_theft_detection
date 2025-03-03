import React from 'react';
import { useStore } from '../ZustandStore/store';
import Login from '../login/Login';
import Registration from '../registration/Registration';

const Modal: React.FC = () => {
  const { isLoginModalOpen, isRegistrationModalOpen, closeModals } = useStore();

  if (!isLoginModalOpen && !isRegistrationModalOpen) return null;

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-gray-900 bg-opacity-50">
      <div className="relative p-4">
        <div className="absolute top-0 right-0 p-2 cursor-pointer" onClick={closeModals}>
          <span className="text-white text-xl">&times;</span>
        </div>
        {isLoginModalOpen && <Login />}
        {isRegistrationModalOpen && <Registration />}
      </div>
    </div>
  );
};

export default Modal;
