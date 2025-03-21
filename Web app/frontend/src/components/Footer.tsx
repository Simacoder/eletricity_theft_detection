import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-blue-800 text-white text-center py-4">
         <p>&copy; {new Date().getFullYear()} Data Phandas Electricity Fraud Detection. All rights reserved.</p>
        <p>
          <a href="/privacy-policy" className="text-blue-400 hover:underline">
            Privacy Policy
          </a> | 
          <a href="/terms" className="text-blue-400 hover:underline">
            Terms of Service
          </a>
        </p>
    </footer>
  );
};

export default Footer;
