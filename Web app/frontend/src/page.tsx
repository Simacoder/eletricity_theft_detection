import Dashboard from '@/pages/Dashboard';
import MeterDataPage from '@/pages/MeterDataPage';
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

const Home = () => {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/meter-data/:meter_id" element={<MeterDataPage />} />
        </Routes>
      </div>
    </Router>
  );
};

export default Home;
