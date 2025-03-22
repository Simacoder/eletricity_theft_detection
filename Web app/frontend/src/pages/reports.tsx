import React, { useEffect } from 'react';
import { useUserStore } from '../store/useUserStore';
import { useRouter } from 'next/router';

type Report = {
  id: string;
  title: string;
  date: string;
};

const Reports: React.FC = () => {
  const { user, isAuthenticated } = useUserStore();
  const router = useRouter();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, router]);

  return (
    <div className="p-6">
      <h1 className="text-3xl font-semibold mb-6">Reports</h1>

      <div>
        <h2 className="text-2xl font-semibold">Recent Reports</h2>
        <table className="w-full mt-4 table-auto">
          <thead>
            <tr>
              <th className="px-4 py-2 border">Report ID</th>
              <th className="px-4 py-2 border">Title</th>
              <th className="px-4 py-2 border">Date</th>
              <th className="px-4 py-2 border">Actions</th>
            </tr>
          </thead>
          <tbody>
            {user && user.reports.map((report: Report) => (
              <tr key={report.id}>
              <td className="px-4 py-2 border">{report.id}</td>
              <td className="px-4 py-2 border">{report.title}</td>
              <td className="px-4 py-2 border">{report.date}</td>
              <td className="px-4 py-2 border">
                <button
                onClick={() => router.push(`/reports/${report.id}`)}
                className="text-blue-500"
                >
                View
                </button>
              </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Reports;
