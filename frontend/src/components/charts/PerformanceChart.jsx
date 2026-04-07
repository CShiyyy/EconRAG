import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

export default function PerformanceChart({ portfolioSeries, benchmarkSeries, initialCash }) {
  const labels = portfolioSeries.map((p) => {
    const d = new Date(p.timestamp);
    return `${d.getMonth() + 1}/${d.getDate()}`;
  });

  const data = {
    labels,
    datasets: [
      {
        label: 'Portfolio',
        data: portfolioSeries.map((p) => p.value),
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37, 99, 235, 0.08)',
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      },
      ...(benchmarkSeries.length > 0
        ? [{
            label: 'SPY Benchmark',
            data: benchmarkSeries.map((b) => b.value),
            borderColor: '#9ca3af',
            borderDash: [5, 3],
            fill: false,
            tension: 0.3,
            pointRadius: 0,
          }]
        : []),
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'top', labels: { boxWidth: 12, font: { size: 12 } } },
      tooltip: {
        callbacks: {
          label: (ctx) =>
            `${ctx.dataset.label}: $${ctx.parsed.y.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
        },
      },
    },
    scales: {
      y: {
        ticks: {
          callback: (v) => `$${(v / 1000).toFixed(0)}k`,
        },
      },
    },
  };

  return (
    <div className="h-80">
      <Line data={data} options={options} />
    </div>
  );
}
