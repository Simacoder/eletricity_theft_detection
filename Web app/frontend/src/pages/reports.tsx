import React, { useEffect, useState } from 'react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import Footer from '../components/Footer';

const ReportsPage: React.FC = () => {
  interface Report {
    id: string;
    title: string;
    date: string;
  }

  const [reports, setReports] = useState<Report[]>([]);

  useEffect(() => {
    // Fetch reports from backend (use your actual API endpoint)
    fetch('/api/reports')
      .then((response) => response.json())
      .then((data) => setReports(data))
      .catch((error) => console.error('Error fetching reports:', error));
  }, []);

  return (
    <div className="flex flex-col h-screen">
      <Navbar />
      <div className="flex flex-1">
        <Sidebar />
        <main className="flex-1 p-6">
          <h1 className="text-3xl font-bold mb-6">Reports</h1>
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Reports Overview</h2>
            <table className="w-full table-auto">
              <thead>
                <tr>
                  <th className="px-4 py-2 border">Report ID</th>
                  <th className="px-4 py-2 border">Title</th>
                  <th className="px-4 py-2 border">Date</th>
                </tr>
              </thead>
              <tbody>
                {reports.length > 0 ? (
                  reports.map((report, index) => (
                    <tr key={index}>
                      <td className="px-4 py-2 border">{report.id}</td>
                      <td className="px-4 py-2 border">{report.title}</td>
                      <td className="px-4 py-2 border">{report.date}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3} className="px-4 py-2 border text-center">No reports available</td>
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

export default ReportsPage;
