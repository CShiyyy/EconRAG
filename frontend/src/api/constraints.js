import api from './client';

export async function getConstraints({ signal } = {}) {
  const { data } = await api.get('/constraints', { signal });
  return data;
}

export async function updateConstraints(constraints) {
  const { data } = await api.patch('/constraints', { constraints });
  return data;
}
