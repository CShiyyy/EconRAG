import api from './client';

export async function getPortfolio({ signal } = {}) {
  const { data } = await api.get('/portfolio', { signal });
  return data;
}

export async function getSnapshots() {
  const { data } = await api.get('/portfolio/snapshots');
  return data;
}

export async function getBenchmark() {
  const { data } = await api.get('/portfolio/benchmark');
  return data;
}
