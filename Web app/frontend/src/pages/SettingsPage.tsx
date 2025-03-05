import { useEffect, useState } from 'react';
import axios from 'axios';

const SettingsPage = () => {
  interface Settings {
    email_notifications: boolean;
    push_notifications: boolean;
  }

  const [settings, setSettings] = useState<Settings>({
    email_notifications: false,
    push_notifications: false,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const response = await axios.get('/api/settings/');
        setSettings(response.data);
      } catch {
        setError('Failed to fetch settings.');
      } finally {
        setLoading(false);
      }
    };

    fetchSettings();
  }, []);

  const handleToggle = async (event: React.ChangeEvent<HTMLInputElement>, setting: string) => {
    const newSettings = { ...settings, [setting]: event.target.checked };
    setSettings(newSettings);

    try {
      await axios.put('/api/settings/', newSettings);
    } catch {
      setError('Failed to update settings.');
    }
  };

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className="settings-page p-4">
      <h1 className="text-2xl font-bold mb-4">Settings</h1>
      
      <section className="mb-6">
        <h2 className="text-xl font-semibold">Notification Preferences</h2>
        
        <div className="flex items-center mb-4">
          <input
            type="checkbox"
            id="email_notifications"
            checked={settings.email_notifications}
            onChange={(e) => handleToggle(e, 'email_notifications')}
            className="mr-2"
          />
          <label htmlFor="email_notifications">Receive email notifications</label>
        </div>
        
        <div className="flex items-center">
          <input
            type="checkbox"
            id="push_notifications"
            checked={settings.push_notifications}
            onChange={(e) => handleToggle(e, 'push_notifications')}
            className="mr-2"
          />
          <label htmlFor="push_notifications">Receive push notifications</label>
        </div>
      </section>
    </div>
  );
};

export default SettingsPage;
