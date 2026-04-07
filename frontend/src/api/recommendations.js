import api from './client';

export async function getRecommendations(skip = 0, limit = 20) {
  const { data } = await api.get('/recommendations', { params: { skip, limit } });
  return data;
}

export async function getRecommendationsByRun(runId) {
  const { data } = await api.get(`/recommendations/${runId}`);
  return data;
}
