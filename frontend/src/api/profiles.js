import api from './client';

export async function getProfile(seedKey) {
  const { data } = await api.get(`/profiles/${seedKey}`);
  return data;
}
