// frontend/src/pages/_app.tsx

import { AppProps } from 'next/app';
import '../styles/globals.css';  // Global styles
import { useEffect } from 'react';
import { useUserStore } from '../store/useUserStore'; // Zustand store for user auth state
import { useRouter } from 'next/router';

function MyApp({ Component, pageProps }: AppProps) {
  const { setUser } = useUserStore();
  const router = useRouter();

  useEffect(() => {
    const userData = localStorage.getItem('user');
    if (userData) {
      setUser(JSON.parse(userData));  // Set the user data in the state
    } else {
      router.push('/login');  // Redirect to login page if no user data is found
    }
  }, []);

  return <Component {...pageProps} />;
}

export default MyApp;
