export const fetchAlerts = async (accessToken: string) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/alerts', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
  
      if (!response.ok) throw new Error('Failed to fetch alerts');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  export const fetchAlertById = async (accessToken: string, alertId: string) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/alerts/${alertId}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
  
      if (!response.ok) throw new Error('Failed to fetch alert');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  export const acknowledgeAlert = async (accessToken: string, alertId: string) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
  
      if (!response.ok) throw new Error('Failed to acknowledge alert');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  export const deleteAlert = async (accessToken: string, alertId: string) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/alerts/${alertId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
  
      if (!response.ok) throw new Error('Failed to delete alert');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  export const createAlert = async (accessToken: string, alertData: { message: string, severity: string }) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/alerts', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(alertData),
      });
  
      if (!response.ok) throw new Error('Failed to create alert');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  