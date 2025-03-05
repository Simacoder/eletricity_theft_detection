import { useEffect, useState } from 'react';
import axios from 'axios';

const HelpPage = () => {
  interface Documentation {
    title: string;
    content: string;
  }

  const [documentation, setDocumentation] = useState<Documentation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDocumentation = async () => {
      try {
        const response = await axios.get('/api/documentation/');
        setDocumentation(response.data);
      } catch (err) {
        console.error(err);
        setError('Failed to fetch documentation.');
      } finally {
        setLoading(false);
      }
    };

    fetchDocumentation();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className="help-page p-4">
      <h1 className="text-2xl font-bold mb-4">Help & Documentation</h1>
      
      {documentation.length === 0 ? (
        <p>No documentation available at the moment.</p>
      ) : (
        documentation.map((doc, index) => (
          <section key={index} className="mb-8">
            <h2 className="text-xl font-semibold mb-2">{doc.title}</h2>
            <p>{doc.content}</p>
          </section>
        ))
      )}
    </div>
  );
};

export default HelpPage;

