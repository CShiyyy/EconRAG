import { useState } from 'react';

export default function JsonViewer({ data, label = 'JSON', defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);

  if (data == null) return <span className="text-gray-400 text-sm">null</span>;

  return (
    <div className="border border-gray-200 rounded">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
      >
        <span>{label}</span>
        <span className="text-xs text-muted">{open ? 'Collapse' : 'Expand'}</span>
      </button>
      {open && (
        <pre className="px-3 py-2 bg-gray-50 text-xs text-gray-700 overflow-x-auto max-h-96 border-t border-gray-200">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}
