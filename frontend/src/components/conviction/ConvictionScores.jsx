// Renders Agent B's top quantitative factor drivers for a recommendation.
// Replaces the legacy narrative_score → multiplier display now that weights
// are deterministic from Agent B and LLMs (Agent C) only annotate.

const CATEGORY_BADGE = {
  Momentum:  'bg-blue-100 text-blue-800',
  Value:     'bg-purple-100 text-purple-800',
  Quality:   'bg-emerald-100 text-emerald-800',
  Technical: 'bg-amber-100 text-amber-800',
  Risk:      'bg-rose-100 text-rose-800',
  Accruals:  'bg-slate-100 text-slate-800',
  Other:     'bg-gray-100 text-gray-700',
};

function FactorChip({ driver }) {
  const z = driver.z_score ?? 0;
  const sign = z >= 0 ? '+' : '';
  const cat = driver.category || 'Other';
  const badge = CATEGORY_BADGE[cat] || CATEGORY_BADGE.Other;
  const zColor = z >= 0 ? 'text-green-700' : 'text-red-700';
  return (
    <span
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${badge}`}
      title={`${driver.factor} (${cat}): cross-sectional Z-score ${sign}${z.toFixed(2)}`}
    >
      <span>{driver.factor}</span>
      <span className={`tabular-nums ${zColor}`}>{sign}{z.toFixed(1)}</span>
    </span>
  );
}

export default function ConvictionScores({ scores }) {
  if (!scores) return <span className="text-gray-400">—</span>;

  const drivers = Array.isArray(scores.factor_drivers) ? scores.factor_drivers : [];
  const composite = scores.composite_signal;

  if (drivers.length === 0) {
    return (
      <span className="text-xs text-gray-400" title="No factor decomposition available for this ticker">
        no factor data
      </span>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap gap-1">
        {drivers.slice(0, 5).map((d, i) => (
          <FactorChip key={`${d.factor}-${i}`} driver={d} />
        ))}
      </div>
      {composite != null && (
        <span
          className="text-[10px] text-gray-500 tabular-nums"
          title="Category-balanced composite signal driving the optimizer"
        >
          composite {composite >= 0 ? '+' : ''}{composite.toFixed(2)}
        </span>
      )}
    </div>
  );
}
