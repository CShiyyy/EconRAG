import api from './client';

export async function getInitStatus() {
  const { data } = await api.get('/init/status');
  return data;
}

export async function postInit(body) {
  const { data } = await api.post('/init', body);
  return data;
}
