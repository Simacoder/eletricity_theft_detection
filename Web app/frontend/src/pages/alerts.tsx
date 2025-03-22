import React, { useEffect } from 'react';
import { useAlertStore } from '../store/useAlertStore';

const Alerts: React.FC = () => {
  const { alerts, isLoading, error, fetchAlerts, acknowledgeAlert, deleteAlert } = useAlertStore();

  useEffect(() => {
    fetchAlerts();
  }, [fetchAlerts]);

  return (
    <div className="p-6">
      <h1 className="text-3xl font-semibold mb-6">Alerts</h1>

      {error && <p className="text-red-500">{error}</p>}

      {isLoading && <p>Loading...</p>}

      <div>
        {alerts.length === 0 ? (
          <p>No alerts available.</p>
        ) : (
          <table className="w-full mt-4 table-auto">
            <thead>
              <tr>
                <th className="px-4 py-2 border">Alert ID</th>
                <th className="px-4 py-2 border">Message</th>
                <th className="px-4 py-2 border">Timestamp</th>
                <th className="px-4 py-2 border">Actions</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((alert) => (
                <tr key={alert.id}>
                  <td className="px-4 py-2 border">{alert.id}</td>
                  <td className="px-4 py-2 border">{alert.message}</td>
                  <td className="px-4 py-2 border">{alert.timestamp}</td>
                  <td className="px-4 py-2 border">
                    {!alert.acknowledged && (
                      <button
                        onClick={() => acknowledgeAlert(alert.id)}
                        className="text-blue-500 mr-2"
                      >
                        Acknowledge
                      </button>
                    )}
                    <button
                      onClick={() => deleteAlert(alert.id)}
                      className="text-red-500"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default Alerts;

