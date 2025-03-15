"use client"
import React from 'react';
import { useRouter } from 'next/navigation';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';

const HomePage: React.FC = () => {
  const router = useRouter();

  const handleLoginClick = () => {
    router.push('/login');
  };

  const handleRegisterClick = () => {
    router.push('/register');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <main className="flex justify-center items-center flex-col py-20">
        <h1 className="text-4xl font-semibold text-center text-blue-600">
          Welcome to the Electricity Fraud Detection System
        </h1>
        <p className="text-xl text-center text-gray-700 mt-4">
          Detect and analyze electricity fraud using our advanced AI-powered system.
        </p>

        <div className="mt-8 flex justify-center gap-8">
          <button
            onClick={handleLoginClick}
            className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Login
          </button>
          <button
            onClick={handleRegisterClick}
            className="px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700"
          >
            Register
          </button>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default HomePage;
