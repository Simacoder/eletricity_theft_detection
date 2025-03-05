import { useEffect, useState } from 'react';
import axios from 'axios';

const AlertsPage = () => {
  interface Alert {
    id: number;
    anomaly: {
      anomaly_type: string;
    };
    alert_severity: string;
    alert_message: string;
    alert_timestamp: string;
  }

  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await axios.get('/api/alerts/');
        setAlerts(response.data);
      } catch {
        setError('Failed to fetch alerts.');
      } finally {
        setLoading(false);
      }
    };

    fetchAlerts();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className="alerts-page p-4">
      <h1 className="text-2xl font-bold mb-4">Alerts</h1>
      
      {alerts.length === 0 ? (
        <p>No alerts to display.</p>
      ) : (
        <table className="min-w-full table-auto mt-2">
          <thead>
            <tr>
              <th className="px-4 py-2 text-left">Anomaly Type</th>
              <th className="px-4 py-2 text-left">Alert Severity</th>
              <th className="px-4 py-2 text-left">Alert Message</th>
              <th className="px-4 py-2 text-left">Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {alerts.map((alert) => (
              <tr key={alert.id}>
                <td className="px-4 py-2">{alert.anomaly.anomaly_type}</td>
                <td className="px-4 py-2">{alert.alert_severity}</td>
                <td className="px-4 py-2">{alert.alert_message}</td>
                <td className="px-4 py-2">{new Date(alert.alert_timestamp).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default AlertsPage;