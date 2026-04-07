import Badge from '../shared/Badge';

const VARIANT_MAP = {
  strong: 'green',
  moderate: 'yellow',
  weak: 'red',
  'n/a': 'gray',
};

export default function ConvictionBadge({ level }) {
  const variant = VARIANT_MAP[level?.toLowerCase()] || 'gray';
  return <Badge variant={variant}>{level || '—'}</Badge>;
}
