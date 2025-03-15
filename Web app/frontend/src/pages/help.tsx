import React from 'react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import Footer from '../components/Footer';

const HelpPage: React.FC = () => {
  return (
    <div className="flex flex-col h-screen">
      <Navbar />
      <div className="flex flex-1">
        <Sidebar />
        <main className="flex-1 p-6">
          <h1 className="text-3xl font-bold mb-6">Help & Documentation</h1>
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">How to Use the Platform</h2>
            <p className="text-sm mb-4">
              Welcome to the Electricity Fraud Detection Platform. Here&apos;s how to use it:
            </p>
            <ul className="list-disc pl-5">
              <li>Register an account to get started</li>
              <li>Monitor your meter data for unusual consumption</li>
              <li>Receive alerts if suspicious activity is detected</li>
              <li>Generate reports to view detailed analytics</li>
            </ul>

            <h2 className="text-xl font-semibold mb-4 mt-6">Frequently Asked Questions</h2>
            <ul className="list-disc pl-5">
              <li>How can I view my meter data? Go to the &quot;Meter Data&quot; page.</li>
              <li>How do I know if my data is suspicious? Check the &quot;Alerts&quot; page for any fraud alerts.</li>
              <li>How do I contact support? Visit the &quot;Help&quot; page for further contact information.</li>
            </ul>

            <h2 className="text-xl font-semibold mb-4 mt-6">Need More Help?</h2>
            <p className="text-sm">
              If you need further assistance, please reach out to our support team at
              support@example.com.
            </p>
          </div>
        </main>
      </div>
      <Footer />
    </div>
  );
};

export default HelpPage;
