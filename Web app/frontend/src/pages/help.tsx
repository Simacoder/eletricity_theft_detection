import React from 'react';

const Help: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-semibold mb-6">Help & Support</h1>

      <div className="space-y-6">
        <section>
          <h2 className="text-2xl font-semibold">How to Use</h2>
          <p className="text-lg">
            To get started, sign up and log in. Once logged in, you can access your dashboard, view
            meter data, check alerts, and generate reports. If you encounter any issues, please refer
            to the troubleshooting section below or reach out to our support team.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold">Frequently Asked Questions</h2>
          <ul className="list-disc pl-6">
            <li>How do I view my meter data?</li>
            <li>What happens if I acknowledge an alert?</li>
            <li>How do I download a report?</li>
          </ul>
        </section>

        <section>
          <h2 className="text-2xl font-semibold">Contact Support</h2>
          <p className="text-lg">
            If you have any further questions or need assistance, please email us at{' '}
            <a href="mailto:support@electricityfraud.com" className="text-blue-500">
              support@electricityfraud.com
            </a>
            .
          </p>
        </section>
      </div>
    </div>
  );
};

export default Help;
