import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-gray-800 text-white py-6 mt-8">
      <div className="max-w-screen-xl mx-auto text-center">
        <p>&copy; 2025 Electricity Detection. All rights reserved.</p>
        <p>
          <a href="/privacy" className="hover:text-gray-400">Privacy Policy</a> | 
          <a href="/terms" className="hover:text-gray-400"> Terms of Service</a>
        </p>
      </div>
    </footer>
  );
};

export default Footer;
