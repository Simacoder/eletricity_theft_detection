export const loginUser = async (email: string, password: string) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });
  
      if (!response.ok) throw new Error('Login failed');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  export const registerUser = async (email: string, password: string, name: string) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password, name }),
      });
  
      if (!response.ok) throw new Error('Registration failed');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  export const logoutUser = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/auth/logout', {
        method: 'POST',
      });
  
      if (!response.ok) throw new Error('Logout failed');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  