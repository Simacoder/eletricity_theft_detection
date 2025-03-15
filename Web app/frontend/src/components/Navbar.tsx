// frontend/src/components/Navbar.tsx

import React from 'react';
import Link from 'next/link';

const Navbar: React.FC = () => {
  return (
    <nav className="bg-gray-800 text-white p-4">
      <div className="flex justify-between items-center">
        <Link href="/" className="text-xl font-bold">Fraud Detection</Link>
        <ul className="flex space-x-6">
          <li><Link href="/" className="hover:text-blue-300">Home</Link></li>
          <li><Link href="/dashboard" className="hover:text-blue-300">Dashboard</Link></li>
          <li><Link href="/meterData" className="hover:text-blue-300">Meter Data</Link></li>
          <li><Link href="/alerts" className="hover:text-blue-300">Alerts</Link></li>
          <li><Link href="/reports" className="hover:text-blue-300">Reports</Link></li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
