import React, { useState } from 'react';
import { useRouter } from 'next/router';
import Button from '../components/Button';

const RegisterPage: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const router = useRouter();

  const handleRegister = async () => {
    if (username && password && email) {
      alert('User registered successfully');
      router.push('/login');
    } else {
      alert('All fields are required!');
    }
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-100">
      <div className="p-8 bg-white rounded-lg shadow-lg w-96">
        <h2 className="text-3xl font-bold mb-4 text-center">Register</h2>
        <div className="mb-4">
          <label className="block text-sm font-semibold mb-2" htmlFor="username">Username</label>
          <input
            type="text"
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full p-3 border rounded-md"
            placeholder="Enter your username"
          />
        </div>
        <div className="mb-4">
          <label className="block text-sm font-semibold mb-2" htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full p-3 border rounded-md"
            placeholder="Enter your email"
          />
        </div>
        <div className="mb-4">
          <label className="block text-sm font-semibold mb-2" htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full p-3 border rounded-md"
            placeholder="Enter your password"
          />
        </div>
        <div className="mt-6">
          <Button text="Register" onClick={handleRegister} />
        </div>
      </div>
    </div>
  );
};

export default RegisterPage;
