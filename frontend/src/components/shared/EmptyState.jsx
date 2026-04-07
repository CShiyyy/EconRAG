export default function EmptyState({ message = 'No data yet' }) {
  return (
    <div className="text-center py-12 text-muted text-sm">
      {message}
    </div>
  );
}
