import api from './client';

export async function getPortfolio() {
  const { data } = await api.get('/portfolio');
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
