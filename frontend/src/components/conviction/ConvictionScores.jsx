export default function ConvictionScores({ scores }) {
  if (!scores) return <span className="text-gray-400">—</span>;

  const ns = scores.narrative_score;
  const multiplier = scores.multiplier;

  if (ns == null) return <span className="text-gray-400">—</span>;

  const nsColor =
    ns >= 7 ? 'text-green-600' :
    ns >= 4 ? 'text-yellow-600' :
    'text-red-600';

  return (
    <span className="text-xs tabular-nums whitespace-nowrap">
      <span
        className={nsColor}
        title={`Narrative score ${ns.toFixed(1)}/10 — Agent C qualitative assessment`}
      >
        {ns.toFixed(1)}
      </span>
      <span className="text-gray-400"> → </span>
      <span
        className="text-gray-700"
        title="Multiplier applied to Agent B quantitative weight (0.5–1.5×)"
      >
        {multiplier != null ? `${multiplier.toFixed(2)}×` : '—'}
      </span>
    </span>
  );
}
