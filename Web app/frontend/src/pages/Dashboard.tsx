import React from 'react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import Footer from '../components/Footer';

const DashboardPage: React.FC = () => {
  return (
    <div className="flex flex-col h-screen">
      <Navbar />
      <div className="flex flex-1">
        <Sidebar />
        <main className="flex-1 p-6">
          <h1 className="text-3xl font-bold mb-6">Dashboard</h1>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-4 bg-white rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4">Meter Data Overview</h2>
              <p>No data available yet. Connect to your meter to start monitoring.</p>
            </div>
            <div className="p-4 bg-white rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4">Fraud Detection</h2>
              <p>No fraud detected yet. Monitoring in progress.</p>
            </div>
          </div>
        </main>
      </div>
      <Footer />
    </div>
  );
};

export default DashboardPage;
