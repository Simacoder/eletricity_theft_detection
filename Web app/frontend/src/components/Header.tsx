import React from 'react';
import Link from 'next/link';

const Header = () => {
  return (
    <header className="bg-blue-600 text-white p-4 shadow-md">
      <div className="max-w-screen-xl mx-auto flex justify-between items-center">
        {/* Logo or App Name */}
        <div className="text-xl font-bold">
          <Link href="/">Electricity Detection</Link>
        </div>
        
        {/* Navigation links */}
        <nav className="space-x-6">
          <Link href="/dashboard" className="hover:text-gray-200">Dashboard</Link>
          <Link href="/meter-data" className="hover:text-gray-200">Meter Data</Link>
          <Link href="/alerts" className="hover:text-gray-200">Alerts</Link>
          <Link href="/settings" className="hover:text-gray-200">Settings</Link>
        </nav>

        {/* Auth links */}
        <div className="space-x-4">
          <Link href="/login" className="hover:text-gray-200">Login</Link>
          <Link href="/register" className="hover:text-gray-200">Register</Link>
        </div>
      </div>
    </header>
  );
};

export default Header;
