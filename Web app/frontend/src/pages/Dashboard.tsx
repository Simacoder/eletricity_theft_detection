"use client"
import { useState, useEffect } from 'react';
import axios from 'axios';

const Dashboard = () => {
  interface MeterData {
    meter_id: string;
    timestamp: string;
    consumption_pattern: string;
    value: number;
  }

  const [meterData, setMeterData] = useState<MeterData[]>([]);
  interface Alert {
    meter: string;
    anomaly_type: string;
    severity: string;
    timestamp: string;
  }

  const [alerts, setAlerts] = useState<Alert[]>([]);
  interface Notification {
    message: string;
    timestamp: string;
  }

  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const response = await axios.get('/api/dashboard/');
        setMeterData(response.data.meter_data);
        setAlerts(response.data.alerts);
        setNotifications(response.data.notifications);
      } catch {
        setError('Failed to load dashboard data.');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className="dashboard p-4">
      <h1 className="text-2xl font-bold mb-4">Dashboard</h1>

      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-2">Meter Data Overview</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {meterData.map((data) => (
            <div key={data.meter_id} className="bg-white p-4 rounded shadow-md">
              <h3 className="font-semibold">{data.meter_id}</h3>
              <p>Timestamp: {new Date(data.timestamp).toLocaleString()}</p>
              <p>Consumption Pattern: {data.consumption_pattern}</p>
              <p>Reading Value: {data.value}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-2">Recent Alerts</h2>
        <div className="space-y-4">
          {alerts.map((alert) => (
            <div key={alert.meter} className="bg-white p-4 rounded shadow-md">
              <p>Meter ID: {alert.meter}</p>
              <p>Anomaly Type: {alert.anomaly_type}</p>
              <p>Severity: {alert.severity}</p>
              <p>Timestamp: {new Date(alert.timestamp).toLocaleString()}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-2">Notifications</h2>
        <div className="space-y-4">
          {notifications.map((notification) => (
            <div key={notification.timestamp} className="bg-white p-4 rounded shadow-md">
              <p>{notification.message}</p>
              <p>Timestamp: {new Date(notification.timestamp).toLocaleString()}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default Dashboard;
