import { useState, useEffect } from 'react';
import LoadingSpinner from '../components/shared/LoadingSpinner';
import { getConstraints, updateConstraints } from '../api/constraints';

export default function ConstraintsPage() {
  const [constraints, setConstraints] = useState([]);
  const [values, setValues] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    getConstraints()
      .then((data) => {
        setConstraints(data.constraints);
        const vals = {};
        data.constraints.forEach((c) => { vals[c.constraint_name] = c.value; });
        setValues(vals);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setSaved(false);
    try {
      const data = await updateConstraints(values);
      setConstraints(data.constraints);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch {
      // stay on page
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <LoadingSpinner />;

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-bold text-gray-900">Portfolio Constraints</h1>

      <div className="bg-white rounded-lg border border-gray-200 p-6 max-w-lg">
        <div className="space-y-4">
          {constraints.map((c) => (
            <div key={c.constraint_name}>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {c.description || c.constraint_name}
              </label>
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={values[c.constraint_name] ?? ''}
                onChange={(e) => setValues({ ...values, [c.constraint_name]: parseFloat(e.target.value) || 0 })}
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          ))}
        </div>

        <div className="mt-6 flex items-center gap-3">
          <button
            onClick={handleSave}
            disabled={saving}
            className="bg-primary text-white text-sm px-5 py-2 rounded hover:bg-primary-dark disabled:opacity-50 transition-colors"
          >
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
          {saved && <span className="text-sm text-success">Saved successfully</span>}
        </div>

        <p className="mt-4 text-xs text-muted">
          Changes take effect on the next pipeline run.
        </p>
      </div>
    </div>
  );
}
