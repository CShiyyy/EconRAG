import ConvictionBadge from './ConvictionBadge';

const LABELS = {
  narrative_alignment: 'Narrative Alignment',
  quant_support: 'Quant Support',
  signal_agreement: 'Signal Agreement',
};

const TOOLTIPS = {
  narrative_alignment: 'How well news, social, and knowledge graph sentiment aligns with a bullish thesis',
  quant_support: 'Degree of support from quantitative metrics: drift, volatility, and health score',
  signal_agreement: 'Agreement between the narrative and quantitative signals',
};

export default function ConvictionScores({ scores }) {
  if (!scores) return <span className="text-gray-400">—</span>;

  const dims = [
    ['narrative_alignment', scores.narrative_alignment, scores.narrative_confidence],
    ['quant_support', scores.quant_support, scores.quant_confidence],
    ['signal_agreement', scores.signal_agreement, scores.signal_confidence],
  ].filter(([, level]) => level && level !== 'n/a');

  return (
    <div className="flex gap-2 flex-wrap">
      {dims.map(([key, level, confidence]) => (
        <span
          key={key}
          className="text-xs text-gray-500 cursor-help"
          title={`${LABELS[key]}: ${level}${confidence != null ? ` (confidence ${confidence.toFixed(1)}/10)` : ''} — ${TOOLTIPS[key]}`}
        >
          {LABELS[key].split(' ').map(w => w[0]).join('')}:{' '}
          <ConvictionBadge level={level} />
          {confidence != null && (
            <span className="ml-0.5 text-gray-400">{confidence.toFixed(1)}</span>
          )}
        </span>
      ))}
    </div>
  );
}
