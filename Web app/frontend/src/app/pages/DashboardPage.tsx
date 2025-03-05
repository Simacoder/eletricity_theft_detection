import { useEffect, useState } from 'react';
import axios from 'axios';

const DashboardPage = () => {
  const [meters, setMeters] = useState([]);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    async function fetchData() {
      const meterResponse = await axios.get('/api/meters/');
      setMeters(meterResponse.data);

      const alertResponse = await axios.get('/api/alerts/');
      setAlerts(alertResponse.data);
    }

    fetchData();
  }, []);

  return (
    <div className="dashboard-page">
      <h2>Dashboard</h2>
      <h3>Meters</h3>
      <ul>
        {meters.map((meter) => (
          <li key={meter.meter_id}>{meter.meter_id}</li>
        ))}
      </ul>

      <h3>Alerts</h3>
      <ul>
        {alerts.map((alert) => (
          <li key={alert.id}>{alert.message}</li>
        ))}
      </ul>
    </div>
  );
};

export default DashboardPage;
