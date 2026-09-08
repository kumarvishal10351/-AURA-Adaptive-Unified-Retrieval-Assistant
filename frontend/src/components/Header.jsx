import React from 'react';

export default function Header({ docCount = 1, onOpenUpload }) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-[#ffffff]/95 backdrop-blur-md border-b border-[#bfc9c1]/30">
      <div className="h-16 w-full px-8 flex items-center justify-between gap-6">
        <div className="flex items-center gap-6 min-w-0 flex-shrink-0">
          <div className="flex items-baseline gap-2">
            <span className="font-['EB_Garamond',serif] text-2xl tracking-tight text-[#005239] font-semibold">
              VIORA
            </span>
            <span className="hidden lg:inline-block font-['Manrope',sans-serif] text-[11px] uppercase tracking-wider text-[#6f7973] border-l border-[#bfc9c1]/50 pl-2">
              DOCUMENT RESEARCH WORKSPACE
            </span>
          </div>
        </div>

        <nav className="hidden md:flex items-center gap-6 h-full">
          <a
            aria-current="page"
            className="h-full flex items-center text-[#005239] border-b-2 border-[#005239] font-semibold text-[0.9375rem] transition-colors"
            href="#"
          >
            Research Workspace
          </a>
          <a
            className="h-full flex items-center text-[#3f4943] hover:text-[#1b1c1e] font-semibold text-[0.8125rem] tracking-wide transition-colors"
            href="#"
            onClick={(e) => { e.preventDefault(); onOpenUpload(); }}
          >
            Documents ({docCount})
          </a>
          <a
            className="h-full flex items-center text-[#3f4943] hover:text-[#1b1c1e] font-semibold text-[0.8125rem] tracking-wide transition-colors"
            href="#"
          >
            Diagnostics
          </a>
        </nav>

        <div className="flex items-center gap-3 flex-shrink-0">
          <div className="hidden xl:flex items-center gap-1.5 px-2.5 py-1 bg-[#c8eadd]/30 border border-[#46645a]/20 rounded text-[#46645a]">
            <span className="material-symbols-outlined text-[15px]">verified</span>
            <span className="font-mono text-[11px] font-semibold tracking-tight">
              Calibrated Grounding: Active
            </span>
          </div>
          <button
            onClick={onOpenUpload}
            className="flex items-center gap-1.5 px-3.5 py-1.5 bg-[#005239] text-white rounded font-semibold text-xs hover:bg-[#1f6b4f] transition-all shadow-sm cursor-pointer"
            type="button"
          >
            <span className="material-symbols-outlined text-[16px]">attach_file</span>
            <span>Attach Document</span>
          </button>
          <div className="w-8 h-8 rounded-full bg-[#005239] flex items-center justify-center flex-shrink-0 ml-1 text-white shadow-sm">
            <span className="material-symbols-outlined text-[18px]">person</span>
          </div>
        </div>
      </div>
    </header>
  );
}
