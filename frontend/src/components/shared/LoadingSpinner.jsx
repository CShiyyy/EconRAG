export default function LoadingSpinner({ size = 'md' }) {
  const dims = size === 'sm' ? 'h-5 w-5' : size === 'lg' ? 'h-12 w-12' : 'h-8 w-8';
  return (
    <div className="flex items-center justify-center p-8">
      <div className={`${dims} animate-spin rounded-full border-4 border-gray-200 border-t-primary`} />
    </div>
  );
}
