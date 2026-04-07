import ConvictionBadge from './ConvictionBadge';

export default function ConvictionScores({ scores }) {
  if (!scores) return <span className="text-gray-400">—</span>;

  return (
    <div className="flex gap-1 flex-wrap">
      {scores.narrative_alignment && (
        <span className="text-xs text-gray-500">
          NA: <ConvictionBadge level={scores.narrative_alignment} />
        </span>
      )}
      {scores.quant_support && (
        <span className="text-xs text-gray-500">
          QS: <ConvictionBadge level={scores.quant_support} />
        </span>
      )}
      {scores.signal_agreement && (
        <span className="text-xs text-gray-500">
          SA: <ConvictionBadge level={scores.signal_agreement} />
        </span>
      )}
    </div>
  );
}
