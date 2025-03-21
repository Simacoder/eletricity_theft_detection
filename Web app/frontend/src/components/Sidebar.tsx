import React from 'react';
import Link from 'next/link';

const Sidebar: React.FC = () => {
  return (
    <div className="w-64 bg-blue-800 text-white p-6 hidden lg:flex flex-col">
      <h2 className="text-xl font-bold mb-6">Electricity Fraud Detection</h2>
      <nav>
        <ul>
          <li>
            <Link href="/" className="block py-2 px-4 hover:bg-blue-700 rounded">
              Home
            </Link>
          </li>
          <li>
            <Link href="/meter-data" className="block py-2 px-4 hover:bg-blue-700 rounded">
              Meter Data
            </Link>
          </li>
          <li>
            <Link href="/reports" className="block py-2 px-4 hover:bg-blue-700 rounded">
              Reports
            </Link>
          </li>
          <li>
            <Link href="/help" className="block py-2 px-4 hover:bg-blue-700 rounded">
              Help
            </Link>
          </li>
          <li>
            <Link href="/settings" className="block py-2 px-4 hover:bg-blue-700 rounded">
              Settings
            </Link>
          </li>
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
