import api from './client';

export async function getConstraints() {
  const { data } = await api.get('/constraints');
  return data;
}

export async function updateConstraints(constraints) {
  const { data } = await api.patch('/constraints', { constraints });
  return data;
}
