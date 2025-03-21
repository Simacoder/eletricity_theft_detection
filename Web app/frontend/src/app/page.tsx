"use client"
import React from 'react';
import Layout from '@/components/Layout';
import Button from '@/components/Button';
import { useRouter } from 'next/navigation';

const Home: React.FC = () => {
  const router = useRouter();

  const handleLoginClick = () => {
    router.push('/login');
  };

  const handleRegisterClick = () => {
    router.push('/register');
  };

  return (
    <Layout>
      <div className="py-20 px-4 sm:px-8 md:px-16">
        <h1 className="text-4xl font-semibold text-center text-blue-600 sm:text-5xl">
          Welcome to the Electricity Fraud Detection System
        </h1>
        <p className="text-xl text-center text-gray-700 mt-4 sm:text-2xl">
          Detect and analyze electricity fraud using our advanced AI-powered system.
        </p>

        <div className="mt-8 flex justify-center gap-8">
          <Button
            text="Login"
            onClick={handleLoginClick}
            variant="primary"
          />
          <Button
            text="Register"
            onClick={handleRegisterClick}
            variant="secondary"
          />
        </div>
      </div>
    </Layout>
  );
};

export default Home;
