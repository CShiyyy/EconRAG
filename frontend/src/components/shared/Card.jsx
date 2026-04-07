export default function Card({ title, value, subtitle, className = '' }) {
  return (
    <div className={`bg-white rounded-lg border border-gray-200 p-4 ${className}`}>
      {title && <p className="text-xs font-medium text-muted uppercase tracking-wide">{title}</p>}
      {value !== undefined && <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>}
      {subtitle && <p className="text-xs text-muted mt-1">{subtitle}</p>}
    </div>
  );
}
