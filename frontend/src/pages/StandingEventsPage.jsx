import { useState, useEffect } from 'react';
import LoadingSpinner from '../components/shared/LoadingSpinner';
import EmptyState from '../components/shared/EmptyState';
import Badge from '../components/shared/Badge';
import Modal from '../components/shared/Modal';
import { getStandingEvents, createStandingEvent, updateStandingEvent } from '../api/standingEvents';

const CATEGORIES = ['geopolitical', 'monetary_policy', 'regulatory', 'trade_policy', 'sector_crisis', 'other'];
const CAT_VARIANT = { geopolitical: 'red', monetary_policy: 'blue', regulatory: 'yellow', trade_policy: 'yellow', sector_crisis: 'red', other: 'gray' };

const EMPTY_FORM = { canonical_id: '', category: 'other', summary: '', affected_tickers: '', stale_run_threshold: 28 };

export default function StandingEventsPage() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showResolved, setShowResolved] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [saving, setSaving] = useState(false);

  const fetchEvents = () => {
    setLoading(true);
    getStandingEvents()
      .then((data) => setEvents(data.items))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchEvents(); }, []);

  const active = events.filter((e) => e.status === 'active');
  const resolved = events.filter((e) => e.status === 'resolved');

  const openCreate = () => {
    setEditingId(null);
    setForm(EMPTY_FORM);
    setModalOpen(true);
  };

  const openEdit = (ev) => {
    setEditingId(ev.standing_id);
    setForm({
      canonical_id: ev.canonical_id,
      category: ev.category,
      summary: ev.summary,
      affected_tickers: Array.isArray(ev.affected_tickers) ? ev.affected_tickers.join(', ') : ev.affected_tickers || '',
      stale_run_threshold: ev.stale_run_threshold || 28,
    });
    setModalOpen(true);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const tickers = form.affected_tickers.split(',').map((t) => t.trim()).filter(Boolean);
      if (editingId) {
        await updateStandingEvent(editingId, {
          summary: form.summary,
          affected_tickers: tickers,
        });
      } else {
        await createStandingEvent({
          canonical_id: form.canonical_id,
          category: form.category,
          summary: form.summary,
          affected_tickers: tickers,
          stale_run_threshold: form.stale_run_threshold,
        });
      }
      setModalOpen(false);
      fetchEvents();
    } catch {
      // keep modal open
    } finally {
      setSaving(false);
    }
  };

  const handleResolve = async (id) => {
    await updateStandingEvent(id, { status: 'resolved' });
    fetchEvents();
  };

  if (loading) return <LoadingSpinner />;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-900">Standing Events</h1>
        <button onClick={openCreate} className="bg-primary text-white text-sm px-4 py-1.5 rounded hover:bg-primary-dark transition-colors">
          Pin New Event
        </button>
      </div>

      {active.length === 0 ? (
        <EmptyState message="No active standing events" />
      ) : (
        <div className="bg-white rounded-lg border border-gray-200 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-3 font-medium text-gray-600">ID</th>
                <th className="text-left py-2 px-3 font-medium text-gray-600">Category</th>
                <th className="text-left py-2 px-3 font-medium text-gray-600">Summary</th>
                <th className="text-left py-2 px-3 font-medium text-gray-600">Tickers</th>
                <th className="text-left py-2 px-3 font-medium text-gray-600">Source</th>
                <th className="text-right py-2 px-3 font-medium text-gray-600">Reinforced</th>
                <th className="text-left py-2 px-3 font-medium text-gray-600">Actions</th>
              </tr>
            </thead>
            <tbody>
              {active.map((ev) => (
                <tr key={ev.standing_id} className="border-b border-gray-50">
                  <td className="py-2 px-3 font-medium text-gray-900">{ev.canonical_id}</td>
                  <td className="py-2 px-3"><Badge variant={CAT_VARIANT[ev.category] || 'gray'}>{ev.category}</Badge></td>
                  <td className="py-2 px-3 text-gray-700 max-w-xs truncate">{ev.summary}</td>
                  <td className="py-2 px-3 text-gray-600 text-xs">
                    {Array.isArray(ev.affected_tickers) ? ev.affected_tickers.join(', ') : ev.affected_tickers}
                  </td>
                  <td className="py-2 px-3 text-gray-500">{ev.promotion_source}</td>
                  <td className="py-2 px-3 text-right text-gray-600">{ev.reinforcement_count ?? 0}x</td>
                  <td className="py-2 px-3">
                    <div className="flex gap-2">
                      <button onClick={() => openEdit(ev)} className="text-primary text-xs hover:underline">Edit</button>
                      <button onClick={() => handleResolve(ev.standing_id)} className="text-danger text-xs hover:underline">Resolve</button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {resolved.length > 0 && (
        <div>
          <button
            onClick={() => setShowResolved(!showResolved)}
            className="text-sm text-muted hover:text-gray-700 flex items-center gap-1"
          >
            <span className={`transition-transform ${showResolved ? 'rotate-90' : ''}`}>&#9654;</span>
            Resolved Events ({resolved.length})
          </button>
          {showResolved && (
            <div className="mt-2 bg-white rounded-lg border border-gray-200 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 px-3 font-medium text-gray-600">ID</th>
                    <th className="text-left py-2 px-3 font-medium text-gray-600">Category</th>
                    <th className="text-left py-2 px-3 font-medium text-gray-600">Summary</th>
                    <th className="text-left py-2 px-3 font-medium text-gray-600">Resolved At</th>
                  </tr>
                </thead>
                <tbody>
                  {resolved.map((ev) => (
                    <tr key={ev.standing_id} className="border-b border-gray-50 text-gray-500">
                      <td className="py-2 px-3">{ev.canonical_id}</td>
                      <td className="py-2 px-3"><Badge variant="gray">{ev.category}</Badge></td>
                      <td className="py-2 px-3 max-w-xs truncate">{ev.summary}</td>
                      <td className="py-2 px-3">{ev.resolved_at || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      <Modal open={modalOpen} onClose={() => setModalOpen(false)} title={editingId ? 'Edit Standing Event' : 'Pin New Standing Event'}>
        <div className="space-y-4">
          {!editingId && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Canonical ID</label>
                <input
                  type="text"
                  value={form.canonical_id}
                  onChange={(e) => setForm({ ...form, canonical_id: e.target.value })}
                  placeholder="e.g. FED_RATE_HIKE_2026"
                  className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
                <select
                  value={form.category}
                  onChange={(e) => setForm({ ...form, category: e.target.value })}
                  className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                >
                  {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            </>
          )}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Summary</label>
            <textarea
              value={form.summary}
              onChange={(e) => setForm({ ...form, summary: e.target.value })}
              rows={3}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Affected Tickers (comma-separated)</label>
            <input
              type="text"
              value={form.affected_tickers}
              onChange={(e) => setForm({ ...form, affected_tickers: e.target.value })}
              placeholder="AAPL, MSFT, NVDA"
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
            />
          </div>
          {!editingId && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Stale Run Threshold</label>
              <input
                type="number"
                value={form.stale_run_threshold}
                onChange={(e) => setForm({ ...form, stale_run_threshold: parseInt(e.target.value) || 28 })}
                className="w-32 border border-gray-300 rounded px-3 py-2 text-sm"
              />
            </div>
          )}
          <div className="flex justify-end gap-2 pt-2">
            <button onClick={() => setModalOpen(false)} className="px-4 py-2 text-sm border rounded hover:bg-gray-50">Cancel</button>
            <button
              onClick={handleSave}
              disabled={saving}
              className="px-4 py-2 text-sm bg-primary text-white rounded hover:bg-primary-dark disabled:opacity-50"
            >
              {saving ? 'Saving...' : editingId ? 'Update' : 'Create'}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
