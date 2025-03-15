export const fetchSettings = async (accessToken: string) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/settings', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
  
      if (!response.ok) throw new Error('Failed to fetch settings');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  export const updateSettings = async (accessToken: string, updatedSettings: { [key: string]: string | number | boolean }) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/settings', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedSettings),
      });
  
      if (!response.ok) throw new Error('Failed to update settings');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  export const resetSettings = async (accessToken: string) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/settings/reset', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
  
      if (!response.ok) throw new Error('Failed to reset settings');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  