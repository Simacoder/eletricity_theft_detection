import React, { useEffect } from 'react';
import { useUserStore } from '../store/useUserStore';
import Layout from '../components/Layout';

import type { AppProps } from 'next/app';

function App({ Component, pageProps }: AppProps) {
  const { user, fetchUser, isAuthenticated } = useUserStore();
 

  useEffect(() => {
    if (!user && !isAuthenticated) {
      fetchUser();
    }
  }, [user, isAuthenticated, fetchUser]);

  return (
    <Layout>
      <Component {...pageProps} />
    </Layout>
  );
}

export default App;
