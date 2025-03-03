import React from 'react';
import { useStore } from '../ZustandStore/store';
import Modal from '../modal/Modal';

const Home: React.FC = () => {
  const { openLoginModal, openRegistrationModal } = useStore();

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="space-x-4">
        <button
          onClick={openLoginModal}
          className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
        >
          Login
        </button>
        <button
          onClick={openRegistrationModal}
          className="bg-green-500 text-white px-4 py-2 rounded-md hover:bg-green-600"
        >
          Register
        </button>
      </div>

      <Modal />
    </div>
  );
};

export default Home;
