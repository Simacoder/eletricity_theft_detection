import React from 'react';
import Link from 'next/link';

const Sidebar = () => {
  return (
    <aside className="bg-gray-800 text-white w-64 p-4 h-full fixed top-0 left-0 flex flex-col space-y-6">
      {/* Logo or App Name */}
      <div className="text-2xl font-bold mb-6">
        <Link href="/">Electricity Detection</Link>
      </div>
      
      {/* Sidebar Links */}
      <nav className="space-y-4">
        <Link href="/dashboard" className="block py-2 px-4 rounded-md hover:bg-gray-700">Dashboard</Link>
        <Link href="/meter-data" className="block py-2 px-4 rounded-md hover:bg-gray-700">Meter Data</Link>
        <Link href="/alerts" className="block py-2 px-4 rounded-md hover:bg-gray-700">Alerts</Link>
        <Link href="/settings" className="block py-2 px-4 rounded-md hover:bg-gray-700">Settings</Link>
        <Link href="/help" className="block py-2 px-4 rounded-md hover:bg-gray-700">Help</Link>
      </nav>
    </aside>
  );
};

export default Sidebar;
