import api from './client';

export async function getRecommendations(skip = 0, limit = 20) {
  const { data } = await api.get('/recommendations', { params: { skip, limit } });
  return data;
}

export async function getLatestRunRecommendations() {
  const { data } = await api.get('/recommendations/latest-run');
  return data;
}

export async function getRecommendationsForDate(sessionDate) {
  const { data } = await api.get('/recommendations', {
    params: { session_date: sessionDate },
  });
  return data;
}

export async function getRecommendationDates() {
  const { data } = await api.get('/recommendations/dates');
  return data;
}

export async function getRecommendationsByRun(runId) {
  const { data } = await api.get(`/recommendations/${runId}`);
  return data;
}
