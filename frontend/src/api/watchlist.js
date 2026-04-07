import api from './client';

export async function getWatchlist() {
  const { data } = await api.get('/watchlist');
  return data;
}

export async function refreshWatchlist() {
  const { data } = await api.post('/watchlist/refresh');
  return data;
}
