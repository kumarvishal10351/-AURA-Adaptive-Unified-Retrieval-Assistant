import React from 'react';

export default function TelemetryBar({ queries = 0, avgConf = 94.2, onSwitchTarget, onIndexFile }) {
  return (
    <section className="w-full bg-[#f5f3f5] px-8 py-2 shadow-sm border-b border-[#bfc9c1]/30 font-mono text-[11px] font-medium text-[#3f4943]">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-1.5">
          <div className="flex items-center gap-1.5 px-2.5 py-1 bg-white rounded shadow-sm">
            <span className="w-2 h-2 rounded-full bg-[#005239] dot-pulse"></span>
            <span className="text-[#005239] font-semibold">Grounded Ready</span>
          </div>

          <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 bg-white rounded shadow-sm">
            <span className="text-[#6f7973]">Queries:</span>
            <span className="font-semibold text-[#1b1c1e]">{queries}</span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 bg-[#c8eadd]/50 text-[#4c6b60] rounded shadow-sm">
            <span className="material-symbols-outlined text-[14px]">bolt</span>
            <span>Avg Conf:</span>
            <span className="font-semibold">{avgConf}%</span>
          </div>

          <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 bg-white rounded shadow-sm">
            <span className="text-[#6f7973]">Cosine Floor:</span>
            <span className="font-semibold text-[#1b1c1e]">≥0.20</span>
          </div>
        </div>

        <div className="flex items-center gap-2 ml-auto font-sans font-semibold text-[11px]">
          <button
            onClick={onSwitchTarget}
            className="flex items-center gap-1 px-2.5 py-1 bg-white text-[#1b1c1e] hover:bg-[#e9e8ea] rounded transition-all shadow-sm cursor-pointer"
            type="button"
          >
            <span className="material-symbols-outlined text-[15px]">swap_horiz</span>
            <span>Switch Target</span>
          </button>
          <button
            onClick={onIndexFile}
            className="flex items-center gap-1 px-2.5 py-1 bg-[#005239] text-white hover:bg-[#1f6b4f] rounded transition-all shadow-sm cursor-pointer"
            type="button"
          >
            <span className="material-symbols-outlined text-[15px]">add</span>
            <span>Index File</span>
          </button>
        </div>
      </div>
    </section>
  );
}
