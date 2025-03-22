import React, { useEffect } from 'react';
import { useUserStore } from '../store/useUserStore';
import { useRouter } from 'next/router';

const Dashboard: React.FC = () => {
  const { user, logout, isAuthenticated } = useUserStore();
  const router = useRouter();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, router]);

  if (!user) {
    return <div>Loading...</div>;
  }

  return (
    <div className="p-6">
      <h1>Welcome, {user.name}</h1>
      <p>Your email: {user.email}</p>
      <p>Your role: {user.role}</p>

      <button
        onClick={() => logout()}
        className="mt-4 px-4 py-2 bg-red-500 text-white rounded"
      >
        Logout
      </button>
    </div>
  );
};

export default Dashboard;

