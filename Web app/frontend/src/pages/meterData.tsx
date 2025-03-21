import React, { useEffect } from 'react';
import { useUserStore } from '../store/useUserStore';
import { useAlertStore } from '../store/useAlertStore';
import { useRouter } from 'next/router';

const MeterData: React.FC = () => {
  const { user, isAuthenticated } = useUserStore();
  const { alerts, fetchAlerts } = useAlertStore();
  const router = useRouter();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login');
    }
    fetchAlerts();
  }, [isAuthenticated, fetchAlerts, router]);

  return (
    <div className="p-6">
      <h1 className="text-3xl font-semibold mb-6">Meter Data</h1>

      <div>
        {alerts.length > 0 && (
          <div className="alert-box bg-yellow-100 text-yellow-700 p-4 rounded mb-4">
            <h3 className="text-xl font-medium">Alerts:</h3>
            <ul>
              {alerts.map((alert) => (
                <li key={alert.id} className="mt-2">
                  {alert.message}
                  <button
                    onClick={() => alert.acknowledged}
                    className="ml-2 text-blue-500"
                  >
                    Acknowledge
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div>
        <h2 className="text-2xl font-semibold">Recent Meter Data</h2>
        <table className="w-full mt-4 table-auto">
          <thead>
            <tr>
              <th className="px-4 py-2 border">Meter ID</th>
              <th className="px-4 py-2 border">Data Type</th>
              <th className="px-4 py-2 border">Value</th>
              <th className="px-4 py-2 border">Timestamp</th>
            </tr>
          </thead>
          <tbody>
          
            {user && user.meterData.map((meter) => (
              <tr key={meter.id}>
                <td className="px-4 py-2 border">{meter.id}</td>
                <td className="px-4 py-2 border">{meter.type}</td>
                <td className="px-4 py-2 border">{meter.value}</td>
                <td className="px-4 py-2 border">{meter.timestamp}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default MeterData;
