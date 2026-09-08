import React from 'react';

export default function QueryComposer({
  query,
  setQuery,
  onSynthesize,
  onSwapSource,
  isLoading
}) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSynthesize();
    }
  };

  return (
    <section className="w-full max-w-4xl mx-auto">
      <div className="bg-white rounded-xl shadow-xl p-6 flex flex-col gap-3 relative overflow-hidden border border-[#bfc9c1]/20">
        {/* Subtle architectural top accent bar */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-[#005239] via-[#46645a] to-[#1f6b4f]"></div>

        {/* Query Input Box */}
        <div className="flex items-start gap-2 pt-1">
          <span className="material-symbols-outlined text-[#005239] text-[26px] mt-1 flex-shrink-0">
            psychology
          </span>
          <div className="flex-1 min-w-0">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              className="w-full bg-[#f5f3f5] text-[#1b1c1e] placeholder:text-[#6f7973] text-[0.9375rem] rounded-lg p-3 focus:outline-none focus:bg-white focus:ring-1 focus:ring-[#005239] transition-all resize-none shadow-sm border border-transparent focus:border-[#005239]/30"
              placeholder="Ask any question about the document corpus (e.g., 'Extract core technical skills and tools')..."
              rows={3}
            />
          </div>
        </div>

        {/* Action Controls & Metadata Row */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-[#efedf0]">
          <div className="flex items-center flex-wrap gap-2">
            <button
              onClick={onSwapSource}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#efedf0] hover:bg-[#e9e8ea] text-[#3f4943] font-semibold text-xs transition-colors shadow-sm cursor-pointer"
              type="button"
            >
              <span className="material-symbols-outlined text-[16px]">attach_file</span>
              <span className="hidden sm:inline">Swap Source</span>
            </button>
            <span className="hidden md:flex items-center gap-1 font-mono text-[11px] text-[#6f7973]">
              <kbd className="px-1.5 py-0.5 bg-[#e9e8ea] rounded text-[#3f4943] shadow-sm text-[11px]">↵ Enter</kbd> to search
            </span>
          </div>

          <div className="flex items-center gap-3 ml-auto">
            <button
              onClick={onSynthesize}
              disabled={isLoading || !query.trim()}
              className="flex items-center gap-1.5 px-6 py-2 bg-[#005239] text-white hover:bg-[#1f6b4f] disabled:opacity-60 rounded-lg font-semibold text-xs transition-all shadow-md group cursor-pointer"
              type="button"
            >
              <span>{isLoading ? 'Searching...' : 'Synthesize'}</span>
              <span className="material-symbols-outlined text-[18px] group-hover:translate-x-0.5 transition-transform">
                arrow_forward
              </span>
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
