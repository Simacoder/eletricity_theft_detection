export const registerUser = async (email: string, password: string, firstName: string, lastName: string) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password, first_name: firstName, last_name: lastName }),
      });
  
      if (!response.ok) throw new Error('Registration failed');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  // Login a user
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
  
  export const fetchUser = async (accessToken: string) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/users/me', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
  
      if (!response.ok) throw new Error('Failed to fetch user details');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  export const updateUser = async (accessToken: string, updatedData: { first_name?: string; last_name?: string; email?: string }) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/users/me', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedData),
      });
  
      if (!response.ok) throw new Error('Failed to update user profile');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  
  export const changePassword = async (accessToken: string, oldPassword: string, newPassword: string) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/users/me/password', {
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ old_password: oldPassword, new_password: newPassword }),
      });
  
      if (!response.ok) throw new Error('Failed to change password');
      return await response.json();
    } catch (error) {
      console.error(error);
      throw error;
    }
  };
  