import api from './client';

export async function getRuns(skip = 0, limit = 20) {
  const { data } = await api.get('/runs', { params: { skip, limit } });
  return data;
}

export async function getRunDetail(runId) {
  const { data } = await api.get(`/runs/${runId}`);
  return data;
}

export async function triggerRun({ overwrite = false } = {}) {
  const { data } = await api.post('/runs/trigger', { overwrite });
  return data;
}

export async function getTriggerStatus(triggerId) {
  const { data } = await api.get(`/runs/trigger/${triggerId}/status`);
  return data;
}

export async function getSlotStatus() {
  const { data } = await api.get('/runs/slot-status');
  return data;
}
