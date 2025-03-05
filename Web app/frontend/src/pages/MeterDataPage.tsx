import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';

const MeterDataPage = () => {
  const { meter_id } = useParams();
  interface MeterData {
    meter_id: string;
    timestamp: string;
    consumption_pattern: string;
    value: number;
  }

  const [meterData, setMeterData] = useState<MeterData | null>(null);
  interface Anomaly {
    anomaly_type: string;
    severity: string;
    timestamp: string;
  }

  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMeterData = async () => {
      try {
        const response = await axios.get(`/api/meter-data/${meter_id}/`);
        setMeterData(response.data.meter_data);
        setAnomalies(response.data.anomalies);
      } catch (err) {
        console.error(err);
        setError('Failed to load meter data.');
      } finally {
        setLoading(false);
      }
    };

    if (meter_id) {
      fetchMeterData();
    }
  }, [meter_id]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className="meter-data-page p-4">
      <h1 className="text-2xl font-bold mb-4">Meter Data - {meterData?.meter_id}</h1>
      <div className="bg-white p-4 rounded shadow-md">
        <h2 className="font-semibold">Meter Information</h2>
        <p>Meter ID: {meterData?.meter_id}</p>
        <p>Timestamp: {new Date(meterData?.timestamp ?? '').toLocaleString()}</p>
        <p>Consumption Pattern: {meterData?.consumption_pattern}</p>
        <p>Reading Value: {meterData?.value}</p>
      </div>

      <section className="mt-8">
        <h2 className="text-xl font-semibold mb-2">Anomalies Detected</h2>
        <div className="space-y-4">
          {anomalies.length > 0 ? (
            anomalies.map((anomaly) => (
              <div key={anomaly.timestamp} className="bg-white p-4 rounded shadow-md">
                <p>Anomaly Type: {anomaly.anomaly_type}</p>
                <p>Severity: {anomaly.severity}</p>
                <p>Timestamp: {new Date(anomaly.timestamp).toLocaleString()}</p>
              </div>
            ))
          ) : (
            <p>No anomalies detected for this meter.</p>
          )}
        </div>
      </section>
    </div>
  );
};

export default MeterDataPage;
