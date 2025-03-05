import { useEffect, useState } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';

const MeterDataPage = () => {
  const { meterId } = useParams<{ meterId: string }>();
  interface MeterData {
    meter: {
      meter_id: string;
      readings: { timestamp: string; value: number }[];
    };
    consumption_patterns: {
      average: number;
      peak: number;
      off_peak: number;
    };
    anomalies: { anomaly_type: string; severity: string; timestamp: string }[];
  }

  const [meterData, setMeterData] = useState<MeterData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMeterData = async () => {
      try {
        const response = await axios.get(`/api/meters/${meterId}/`);
        setMeterData(response.data);
      } catch {
        setError('Failed to fetch meter data.');
      } finally {
        setLoading(false);
      }
    };

    fetchMeterData();
  }, [meterId]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className="meter-data-page p-4">
      <h1 className="text-2xl font-bold mb-4">Meter Data for {meterData?.meter.meter_id}</h1>
      
      <section>
        <h2 className="text-xl font-semibold">Consumption Patterns</h2>
        <div className="grid grid-cols-3 gap-4 mt-2">
          <div className="p-4 border border-gray-300">
            <h3 className="font-medium">Average Consumption</h3>
            <p>{meterData?.consumption_patterns.average} kWh</p>
          </div>
          <div className="p-4 border border-gray-300">
            <h3 className="font-medium">Peak Consumption</h3>
            <p>{meterData?.consumption_patterns.peak} kWh</p>
          </div>
          <div className="p-4 border border-gray-300">
            <h3 className="font-medium">Off-Peak Consumption</h3>
            <p>{meterData?.consumption_patterns.off_peak} kWh</p>
          </div>
        </div>
      </section>

      <section className="mt-6">
        <h2 className="text-xl font-semibold">Meter Readings</h2>
        <table className="min-w-full table-auto mt-2">
          <thead>
            <tr>
              <th className="px-4 py-2 text-left">Timestamp</th>
              <th className="px-4 py-2 text-left">Reading Value (kWh)</th>
            </tr>
          </thead>
          <tbody>
            {meterData?.meter.readings.map((reading: { timestamp: string; value: number }, index: number) => (
              <tr key={index}>
                <td className="px-4 py-2">{reading.timestamp}</td>
                <td className="px-4 py-2">{reading.value} kWh</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="mt-6">
        <h2 className="text-xl font-semibold">Anomalies</h2>
        {meterData?.anomalies.length === 0 ? (
          <p>No anomalies detected.</p>
        ) : (
          <table className="min-w-full table-auto mt-2">
            <thead>
              <tr>
                <th className="px-4 py-2 text-left">Anomaly Type</th>
                <th className="px-4 py-2 text-left">Severity</th>
                <th className="px-4 py-2 text-left">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {meterData?.anomalies.map((anomaly: { anomaly_type: string; severity: string; timestamp: string }, index: number) => (
                <tr key={index}>
                  <td className="px-4 py-2">{anomaly.anomaly_type}</td>
                  <td className="px-4 py-2">{anomaly.severity}</td>
                  <td className="px-4 py-2">{new Date(anomaly.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
};

export default MeterDataPage;
