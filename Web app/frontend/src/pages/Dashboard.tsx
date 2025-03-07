import React, { useEffect } from 'react';
import { useStore } from '../store/zustandStore';

const Dashboard = () => {
  const { user, meterData, alerts, setMeterData, setAlerts } = useStore();

  // Fetch meter data and alerts from API
  const fetchMeterData = async () => {
    try {
      const response = await fetch('/api/meter-data/');
      if (response.ok) {
        const data = await response.json();
        setMeterData(data); 
      } else {
        console.error('Failed to fetch meter data');
      }
    } catch (error) {
      console.error('Error fetching meter data:', error);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await fetch('/api/alerts/');
      if (response.ok) {
        const data = await response.json();
        setAlerts(data);
      } else {
        console.error('Failed to fetch alerts');
      }
    } catch (error) {
      console.error('Error fetching alerts:', error);
    }
  };

  useEffect(() => {
    // Fetch data on component mount
    fetchMeterData();
    fetchAlerts();
  }, [setMeterData, setAlerts]);

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">
        Welcome, {user?.username || 'Guest'}!
      </h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* Meter Data Overview */}
        <div className="bg-white p-6 shadow-lg rounded-md">
          <h2 className="text-2xl font-semibold mb-4">Meter Data Overview</h2>
          <ul>
            {meterData.length > 0 ? (
              meterData.map((data: { meter_id: string, value: number, timestamp: string }) => (
                <li key={data.meter_id} className="mb-4">
                  <strong>Meter {data.meter_id}:</strong> {data.value} kWh at {new Date(data.timestamp).toLocaleString()}
                </li>
              ))
            ) : (
              <p>No meter data available.</p>
            )}
          </ul>
        </div>
        
        {/* Alerts Overview */}
        <div className="bg-white p-6 shadow-lg rounded-md">
          <h2 className="text-2xl font-semibold mb-4">Recent Alerts</h2>
          <ul>
            {alerts.length > 0 ? (
              alerts.map((alert: { id: string, message: string, severity: string }) => (
                <li key={alert.id} className="mb-4">
                  <strong>{alert.message}</strong> <br />
                  <span className={`text-${alert.severity === 'High' ? 'red' : alert.severity === 'Medium' ? 'orange' : 'green'}-500`}>
                    Severity: {alert.severity}
                  </span>
                </li>
              ))
            ) : (
              <p>No recent alerts.</p>
            )}
          </ul>
        </div>
      </div>

      {/* Notifications Section */}
      <div className="bg-white p-6 shadow-lg rounded-md">
        <h2 className="text-2xl font-semibold mb-4">Notifications</h2>
        <p>No new notifications at the moment.</p>
      </div>
    </div>
  );
};

export default Dashboard;
