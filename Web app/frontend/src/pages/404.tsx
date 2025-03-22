import React from 'react';
import Link from 'next/link';

const NotFound: React.FC = () => {
  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50 text-center">
      <div>
        <h1 className="text-4xl font-semibold text-red-600 mb-4">404 - Page Not Found</h1>
        <p className="text-lg text-gray-700 mb-4">Sorry, the page you are looking for does not exist.</p>
        <Link href="/" className="text-blue-500 text-lg hover:underline">
          Go back to Home
        </Link>
      </div>
    </div>
  );
};

export default NotFound;

