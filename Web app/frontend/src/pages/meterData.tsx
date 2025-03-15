import React, { useEffect, useState } from 'react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import Footer from '../components/Footer';

const MeterDataPage: React.FC = () => {
  interface MeterData {
    household_id: string;
    consumption: number;
    date: string;
  }

  const [meterData, setMeterData] = useState<MeterData[]>([]);

  useEffect(() => {
    fetch('http://127.0.0.1:8000//api/meter-data')
      .then((response) => response.json())
      .then((data) => setMeterData(data))
      .catch((error) => console.error('Error fetching meter data:', error));
  }, []);

  return (
    <div className="flex flex-col h-screen">
      <Navbar />
      <div className="flex flex-1">
        <Sidebar />
        <main className="flex-1 p-6">
          <h1 className="text-3xl font-bold mb-6">Meter Data</h1>
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Meter Data Overview</h2>
            <table className="w-full table-auto">
              <thead>
                <tr>
                  <th className="px-4 py-2 border">Household ID</th>
                  <th className="px-4 py-2 border">Consumption</th>
                  <th className="px-4 py-2 border">Date</th>
                </tr>
              </thead>
              <tbody>
                {meterData.length > 0 ? (
                  meterData.map((data, index) => (
                    <tr key={index}>
                      <td className="px-4 py-2 border">{data.household_id}</td>
                      <td className="px-4 py-2 border">{data.consumption}</td>
                      <td className="px-4 py-2 border">{data.date}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3} className="px-4 py-2 border text-center">No data available</td>
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

export default MeterDataPage;
