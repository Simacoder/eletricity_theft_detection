export const fetchMeterData = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/meter_data');
      if (!response.ok) throw new Error('Failed to fetch meter data');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  interface MeterData {
    id: number;
    value: number;
    timestamp: string;
  }

  export const postMeterData = async (data: MeterData) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/meter_data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
  
      if (!response.ok) throw new Error('Failed to post meter data');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  