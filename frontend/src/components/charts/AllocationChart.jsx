import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

const COLORS = [
  '#2563eb', '#7c3aed', '#db2777', '#ea580c', '#16a34a',
  '#0891b2', '#4f46e5', '#c026d3', '#d97706', '#059669',
  '#6366f1', '#e11d48', '#0d9488', '#8b5cf6', '#f59e0b',
];

export default function AllocationChart({ weights }) {
  const entries = Object.entries(weights).filter(([, w]) => w > 0);

  if (entries.length === 0) return null;

  const data = {
    labels: entries.map(([ticker]) => ticker),
    datasets: [{
      data: entries.map(([, w]) => (w * 100).toFixed(2)),
      backgroundColor: entries.map((_, i) => COLORS[i % COLORS.length]),
      borderWidth: 1,
      borderColor: '#fff',
    }],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'right', labels: { boxWidth: 12, font: { size: 11 } } },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.label}: ${ctx.parsed}%`,
        },
      },
    },
  };

  return (
    <div className="h-64">
      <Pie data={data} options={options} />
    </div>
  );
}
