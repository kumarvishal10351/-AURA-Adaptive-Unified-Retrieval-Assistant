import React from 'react';

export default function Footer({ latency = 42 }) {
  return (
    <footer className="w-full border-t border-[#bfc9c1]/30 bg-white py-3.5 px-8 mt-auto">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-[#6f7973] font-semibold text-[11px]">
        <div>
          Viora Research Workbench • Epistemic Grounding Engine • System Integrity 99.98%
        </div>
        <div className="flex items-center gap-6">
          <span>Latency: {latency}ms</span>
          <span>Cosine Threshold: 0.20</span>
          <span>© 2026 Viora Labs</span>
        </div>
      </div>
    </footer>
  );
}
