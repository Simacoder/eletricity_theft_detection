import React, { useEffect, useState } from 'react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import Footer from '../components/Footer';

const AlertsPage: React.FC = () => {
  interface Alert {
    id: string;
    household_id: string;
    alert_type: string;
    date: string;
  }

  const [alerts, setAlerts] = useState<Alert[]>([]);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/alerts')
      .then((response) => response.json())
      .then((data) => setAlerts(data))
      .catch((error) => console.error('Error fetching alerts:', error));
  }, []);

  return (
    <div className="flex flex-col h-screen">
      <Navbar />
      <div className="flex flex-1">
        <Sidebar />
        <main className="flex-1 p-6">
          <h1 className="text-3xl font-bold mb-6">Alerts</h1>
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Alerts Overview</h2>
            <table className="w-full table-auto">
              <thead>
                <tr>
                  <th className="px-4 py-2 border">Alert ID</th>
                  <th className="px-4 py-2 border">Household ID</th>
                  <th className="px-4 py-2 border">Alert Type</th>
                  <th className="px-4 py-2 border">Date</th>
                </tr>
              </thead>
              <tbody>
                {alerts.length > 0 ? (
                  alerts.map((alert, index) => (
                    <tr key={index}>
                      <td className="px-4 py-2 border">{alert.id}</td>
                      <td className="px-4 py-2 border">{alert.household_id}</td>
                      <td className="px-4 py-2 border">{alert.alert_type}</td>
                      <td className="px-4 py-2 border">{alert.date}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={4} className="px-4 py-2 border text-center">No alerts available</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </main>
      </div>
      <Footer />
    </div>
  );
};

export default AlertsPage;
