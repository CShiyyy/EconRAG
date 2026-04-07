import { useState, useEffect, useRef, useCallback } from 'react';

export function usePolling(fetchFn, intervalMs = 2000, enabled = false) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!enabled || !fetchFn) {
      stop();
      return;
    }

    const poll = async () => {
      try {
        const result = await fetchFn();
        setData(result);
        if (result.status === 'completed' || result.status === 'failed') {
          stop();
        }
      } catch (err) {
        setError(err);
        stop();
      }
    };

    poll();
    intervalRef.current = setInterval(poll, intervalMs);

    return stop;
  }, [enabled, fetchFn, intervalMs, stop]);

  return { data, error, stop };
}
