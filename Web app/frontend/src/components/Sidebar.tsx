// frontend/src/components/Sidebar.tsx

import React from 'react';
import Link from 'next/link';

const Sidebar: React.FC = () => {
  return (
    <div className="bg-gray-800 text-white w-64 p-4 h-screen">
      <h2 className="text-2xl font-bold mb-6">Fraud Detection</h2>
      <ul className="space-y-4">
        <li><Link href="/dashboard" className="hover:text-blue-300">Dashboard</Link></li>
        <li><Link href="/meterData" className="hover:text-blue-300">Meter Data</Link></li>
        <li><Link href="/alerts" className="hover:text-blue-300">Alerts</Link></li>
        <li><Link href="/reports" className="hover:text-blue-300">Reports</Link></li>
        <li><Link href="/settings" className="hover:text-blue-300">Settings</Link></li>
        <li><Link href="/help" className="hover:text-blue-300">Help</Link></li>
      </ul>
    </div>
  );
};

export default Sidebar;

