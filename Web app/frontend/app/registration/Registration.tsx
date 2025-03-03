import React from 'react';
import { useStore } from '../ZustandStore/store';

const Registration: React.FC = () => {
  const { closeModals } = useStore();

  return (
    <div className="bg-white p-8 rounded-md shadow-lg w-80">
      <h2 className="text-2xl font-semibold mb-4">Register</h2>
      <form className="space-y-4">
        <input
          type="email"
          placeholder="Email"
          className="w-full p-2 border border-gray-300 rounded-md"
        />
        <input
          type="password"
          placeholder="Password"
          className="w-full p-2 border border-gray-300 rounded-md"
        />
        <input
          type="password"
          placeholder="Confirm Password"
          className="w-full p-2 border border-gray-300 rounded-md"
        />
        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600"
        >
          Register
        </button>
      </form>
      <div className="mt-4 text-center">
        <button
          onClick={closeModals}
          className="text-blue-500 hover:underline"
        >
          Close
        </button>
      </div>
    </div>
  );
};

export default Registration;
