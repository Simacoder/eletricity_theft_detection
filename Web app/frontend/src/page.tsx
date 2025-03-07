import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './Layout/Layout';
import Dashboard from './pages/Dashboard';
import MeterDataPage from './pages/MeterDataPage';
import AlertsPage from './pages/AlertsPage';
import SettingsPage from './pages/SettingsPage';
import HelpPage from './pages/HelpPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';


const Home = () => {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/meter-data/:meter_id" element={<MeterDataPage />} />
          <Route path="/alerts" element={<AlertsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/help" element={<HelpPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
        </Routes>
      </Layout>
    </Router>
  );
};

export default Home;