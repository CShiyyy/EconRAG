import { useState } from 'react';

export default function ExpandableRow({ summary, children }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border-b border-gray-100">
      <button
        onClick={() => setOpen(!open)}
        className="w-full text-left flex items-center gap-1 py-1 text-sm text-primary hover:underline"
      >
        <span className={`transition-transform ${open ? 'rotate-90' : ''}`}>&#9654;</span>
        {summary}
      </button>
      {open && <div className="pb-2 pl-4 text-sm text-gray-600">{children}</div>}
    </div>
  );
}
