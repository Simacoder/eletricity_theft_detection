import React from 'react';
import { useRouter } from 'next/router';

const NotFoundPage: React.FC = () => {
  const router = useRouter();

  const handleGoHome = () => {
    router.push('/');
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-100">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4">404 - Page Not Found</h1>
        <p className="text-lg mb-6">Sorry, the page you&apos;re looking for does not exist.</p>
        <button
          onClick={handleGoHome}
          className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
        >
          Go to Home
        </button>
      </div>
    </div>
  );
};

export default NotFoundPage;
