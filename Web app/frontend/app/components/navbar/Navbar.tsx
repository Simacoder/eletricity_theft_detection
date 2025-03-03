import Image from 'next/image';
import Link from 'next/link';
import React from 'react';
import SearchFilter from './SearchFilter';

const Navbar = () => {
  return (
    <nav className="w-full fixed top-0 left-0 border-b bg-white z-10">
      <div className="max-w-screen-xl mx-auto px-6">
        <div className="flex justify-between items-center py-4">
          <Link legacyBehavior href="/">
            <a>
              <Image src="/Logo.png" alt="logo" width={180} height={30} />
            </a>
          </Link>
          <div>
            <ul className="hidden md:flex space-x-8">
              <li>
                <SearchFilter />
              </li>
              <li>
                <Link href="/dashboard">Dashboard</Link>
              </li>
              <li>
                <Link href="/meter-data">Meter Data</Link>
              </li>
              <li>
                <Link href="/reports">Reports</Link>
              </li>
              <li>
                <Link href="/alerts">Alerts</Link>
              </li>
              <li>
                <Link href="/settings">Settings</Link>
              </li>
              <li>
                <Link href="/help">Help</Link>
              </li>
              <li>
                <Link href="/login">Login/Register</Link>
              </li>
            </ul>
            {/* Mobile Menu */}
            <div className="md:hidden">
              <button className="text-gray-600 focus:outline-none">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  className="w-6 h-6"
                >
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

