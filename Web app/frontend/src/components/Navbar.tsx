import React from 'react';
import Link from 'next/link';

const Navbar: React.FC = () => {
  return (
    <header className="bg-blue-800 text-white p-4 flex justify-between items-center">
      <Link href="/" className="text-2xl font-bold">
        Fraud Detection System
      </Link>
      <nav>
        <ul className="flex space-x-6">
          <li>
            <Link href="/" className="hover:text-blue-300">Home</Link>
          </li>
          <li>
            <Link href="/login" className="hover:text-blue-300">Login</Link>
          </li>
          <li>
            <Link href="/register" className="hover:text-blue-300">Register</Link>
          </li>
        </ul>
      </nav>
    </header>
  );
};

export default Navbar;
