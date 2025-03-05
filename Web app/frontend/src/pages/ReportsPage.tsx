import { useState } from 'react';
import axios from 'axios';

const ReportsPage = () => {
  const [reportType, setReportType] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await axios.post('/api/reports/', {
        report_type: reportType,
      });
      alert('Report generated: ' + response.data.message);
    } catch (error) {
      console.error('Error generating report:', error);
    }
  };

  return (
    <div className="reports-page">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Report Type"
          value={reportType}
          onChange={(e) => setReportType(e.target.value)}
          required
        />
        <button type="submit">Generate Report</button>
      </form>
    </div>
  );
};

export default ReportsPage;
