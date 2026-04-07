import api from './client';

export async function getStandingEvents(status) {
  const params = status ? { status } : {};
  const { data } = await api.get('/standing-events', { params });
  return data;
}

export async function createStandingEvent(body) {
  const { data } = await api.post('/standing-events', body);
  return data;
}

export async function updateStandingEvent(id, body) {
  const { data } = await api.patch(`/standing-events/${id}`, body);
  return data;
}
