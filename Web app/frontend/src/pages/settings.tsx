import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import Footer from '../components/Footer';

const SettingsPage: React.FC = () => {
  const [userSettings, setUserSettings] = useState({
    emailNotifications: false,
    darkMode: false,
  });

  useEffect(() => {
    fetch('http://127.0.0.1:8000//api/settings')
      .then((response) => response.json())
      .then((data) => setUserSettings(data))
      .catch((error) => console.error('Error fetching settings:', error));
  }, []);

  const handleSettingsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = e.target;
    setUserSettings((prevSettings) => ({
      ...prevSettings,
      [name]: checked,
    }));
  };

  const saveSettings = async () => {
    console.log('Settings saved:', userSettings);
    alert('Settings have been saved!');
  };

  return (
    <div className="flex flex-col h-screen">
      <Navbar />
      <div className="flex flex-1">
        <Sidebar />
        <main className="flex-1 p-6">
          <h1 className="text-3xl font-bold mb-6">Settings</h1>
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Notification Settings</h2>
            <div className="mb-4">
              <label className="block text-sm font-semibold">
                Email Notifications
                <input
                  type="checkbox"
                  name="emailNotifications"
                  checked={userSettings.emailNotifications}
                  onChange={handleSettingsChange}
                  className="ml-2"
                />
              </label>
            </div>
            <h2 className="text-xl font-semibold mb-4">Appearance Settings</h2>
            <div className="mb-4">
              <label className="block text-sm font-semibold">
                Dark Mode
                <input
                  type="checkbox"
                  name="darkMode"
                  checked={userSettings.darkMode}
                  onChange={handleSettingsChange}
                  className="ml-2"
                />
              </label>
            </div>
            <div className="mt-6">
              <button
                onClick={saveSettings}
                className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
              >
                Save Settings
              </button>
            </div>
          </div>
        </main>
      </div>
      <Footer />
    </div>
  );
};

export default SettingsPage;
