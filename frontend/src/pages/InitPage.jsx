import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { postInit } from '../api/init';

const UNIVERSES = [
  { value: 'djia30', label: 'Dow Jones 30' },
  { value: 'nasdaq100', label: 'Nasdaq 100' },
  { value: 'sp500', label: 'S&P 500' },
];

const DEFAULT_CONSTRAINTS = {
  cash_floor: 0.05,
  max_single_position: 0.15,
  max_sector_concentration: 0.35,
  min_position_size: 0.02,
};

const CONSTRAINT_LABELS = {
  cash_floor: 'Cash Floor',
  max_single_position: 'Max Single Position',
  max_sector_concentration: 'Max Sector Concentration',
  min_position_size: 'Min Position Size',
};

export default function InitPage() {
  const navigate = useNavigate();
  const [universe, setUniverse] = useState('djia30');
  const [startingCash, setStartingCash] = useState(100000);
  const [constraints, setConstraints] = useState(DEFAULT_CONSTRAINTS);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleConstraintChange = (key, value) => {
    setConstraints((prev) => ({ ...prev, [key]: parseFloat(value) || 0 }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await postInit({
        universe,
        starting_cash: startingCash,
        constraints,
      });
      navigate('/', { replace: true });
    } catch (err) {
      setError(err.response?.data?.detail || 'Initialization failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-surface flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-lg p-8 w-full max-w-lg">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">EconRAG</h1>
        <p className="text-muted mb-6">Initialize your portfolio monitoring system</p>

        {error && (
          <div className="bg-red-50 border border-red-200 text-danger rounded p-3 mb-4 text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Ticker Universe
            </label>
            <select
              value={universe}
              onChange={(e) => setUniverse(e.target.value)}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            >
              {UNIVERSES.map((u) => (
                <option key={u.value} value={u.value}>{u.label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Starting Cash ($)
            </label>
            <input
              type="number"
              min={1000}
              step={1000}
              value={startingCash}
              onChange={(e) => setStartingCash(parseFloat(e.target.value) || 0)}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          <fieldset className="border border-gray-200 rounded p-4">
            <legend className="text-sm font-medium text-gray-700 px-1">
              Constraints
            </legend>
            <div className="space-y-3 mt-1">
              {Object.entries(constraints).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between gap-4">
                  <label className="text-sm text-gray-600 flex-1">
                    {CONSTRAINT_LABELS[key]}
                  </label>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={value}
                    onChange={(e) => handleConstraintChange(key, e.target.value)}
                    className="w-24 border border-gray-300 rounded px-2 py-1 text-sm text-right focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>
              ))}
            </div>
          </fieldset>

          <button
            type="submit"
            disabled={submitting}
            className="w-full bg-primary text-white rounded py-2.5 text-sm font-medium hover:bg-primary-dark disabled:opacity-50 transition-colors"
          >
            {submitting ? 'Initializing...' : 'Initialize System'}
          </button>
        </form>
      </div>
    </div>
  );
}
