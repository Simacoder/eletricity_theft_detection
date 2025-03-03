"use client"
import React, { useEffect, useState } from 'react';
import axios from 'axios';

const Home: React.FC = () => {
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    axios.get('http://127.0.0.1:8000/api/home/')
      .then(response => {
        setMessage(response.data.message);
      })
      .catch(error => {
        console.error('There was an error!', error);
      });
  }, []);

  return (
    <div className="Home">
      <h1 className='text-green-500 font-bold text-3xl'>{message ? message : 'Loading...'}</h1>
    </div>
  );
}

export default Home;
