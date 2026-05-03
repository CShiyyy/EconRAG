import api from './client';

export async function getTrades({ status, skip = 0, limit = 50 } = {}) {
  const { data } = await api.get('/trades', { params: { status, skip, limit } });
  return data;
}

export async function getLatestRunTrades() {
  const { data } = await api.get('/trades/latest-run');
  return data;
}
