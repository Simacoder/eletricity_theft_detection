import React, { useEffect, useState } from 'react';
import { useUserStore } from '../store/useUserStore';
import { useRouter } from 'next/router';

const Settings: React.FC = () => {
  const { user, isAuthenticated } = useUserStore();
  const [settings, setSettings] = useState({
    emailNotifications: false,
    darkMode: false,
  });
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login');
    }
    if (user) {
      setSettings({
        emailNotifications: user.emailNotifications,
        darkMode: user.darkMode,
      });
    }
  }, [isAuthenticated, user, router]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = e.target;
    setSettings((prev) => ({ ...prev, [name]: checked }));
  };

  const saveSettings = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings),
      });
      if (!response.ok) throw new Error('Failed to save settings');
    } catch (error) {
      console.error('Error saving settings:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-3xl font-semibold mb-6">Settings</h1>

      {loading && <p>Saving settings...</p>}

      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-semibold mb-4">Notification Settings</h2>
          <label className="block text-sm font-semibold">
            <input
              type="checkbox"
              name="emailNotifications"
              checked={settings.emailNotifications}
              onChange={handleChange}
              className="mr-2"
            />
            Email Notifications
          </label>
        </div>

        <div>
          <h2 className="text-2xl font-semibold mb-4">Appearance Settings</h2>
          <label className="block text-sm font-semibold">
            <input
              type="checkbox"
              name="darkMode"
              checked={settings.darkMode}
              onChange={handleChange}
              className="mr-2"
            />
            Dark Mode
          </label>
        </div>

        <div>
          <button
            onClick={saveSettings}
            className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            disabled={loading}
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;
