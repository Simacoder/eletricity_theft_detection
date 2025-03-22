const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

interface User {
  id: string;
  email: string;
  name: string;
}

export const loginUser = async (email: string, password: string): Promise<User> => {
  try {
    const response = await fetch(`${API_URL}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const errorDetails = await response.json();
      throw new Error(`Error ${response.status}: ${errorDetails.message || 'Login failed'}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Login Error:', error);
    throw error;
  }
};

export const registerUser = async (email: string, password: string, name: string): Promise<User> => {
  try {
    const response = await fetch(`${API_URL}/api/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password, name }),
    });

    if (!response.ok) {
      const errorDetails = await response.json();
      throw new Error(`Error ${response.status}: ${errorDetails.message || 'Registration failed'}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Registration Error:', error);
    throw error;
  }
};

export const logoutUser = async (token: string) => {
  try {
    const response = await fetch(`${API_URL}/api/auth/logout`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      const errorDetails = await response.json();
      throw new Error(`Error ${response.status}: ${errorDetails.message || 'Logout failed'}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Logout Error:', error);
    throw error;
  }
};
