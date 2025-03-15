export const fetchReports = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/reports');
      if (!response.ok) throw new Error('Failed to fetch reports');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  interface ReportData {
    title: string;
    content: string;
  }

  export const generateReport = async (reportData: ReportData) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/reports', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(reportData),
      });
  
      if (!response.ok) throw new Error('Failed to generate report');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  ;
  