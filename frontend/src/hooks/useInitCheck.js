import { useState, useEffect } from 'react';
import { getInitStatus } from '../api/init';

export function useInitCheck() {
  const [initialized, setInitialized] = useState(false);
  const [account, setAccount] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getInitStatus()
      .then((data) => {
        setInitialized(data.initialized);
        setAccount(data.account);
      })
      .catch(() => setInitialized(false))
      .finally(() => setLoading(false));
  }, []);

  return { initialized, account, loading };
}
