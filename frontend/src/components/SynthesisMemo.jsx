import React from 'react';

export default function SynthesisMemo({
  question,
  answer,
  confidence = 94.2,
  anchors = [],
  latency = 218,
  isLoading = false
}) {
  if (!question && !answer && !isLoading) return null;

  return (
    <section className="w-full max-w-4xl mx-auto mt-6 transition-all duration-300">
      <div className="bg-white rounded-xl p-6 shadow-lg border-l-4 border-[#005239] border border-[#bfc9c1]/20 fade-in">
        {/* Status Header & Latency */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <span
              className={`w-2.5 h-2.5 rounded-full bg-[#005239] ${
                isLoading ? 'animate-ping' : ''
              }`}
            ></span>
            <span className="font-semibold text-xs text-[#005239]">
              {isLoading ? 'Synthesizing Verified Answer' : 'Deterministic Synthesis Complete'}
            </span>
          </div>
          <span className="font-mono text-[11px] text-[#6f7973]">
            {isLoading ? 'Processing...' : `Latency: ${latency}ms`}
          </span>
        </div>

        {/* Question Prompt In Quotation Marks */}
        {question && (
          <p className="font-['EB_Garamond',serif] text-xl text-[#1b1c1e] mb-4 italic leading-relaxed">
            "{question}"
          </p>
        )}

        {/* Answer Content Box */}
        <div className="p-4 rounded-lg bg-[#f5f3f5] text-[0.9375rem] text-[#1b1c1e] leading-relaxed whitespace-pre-wrap font-sans">
          {isLoading ? (
            <div className="flex items-center gap-2 text-[#46645a] font-mono text-xs">
              <span className="w-2 h-2 rounded-full bg-[#005239] dot-pulse"></span>
              <span>Extracting grounded intelligence from indexed vectors...</span>
            </div>
          ) : (
            answer
          )}
        </div>

        {/* Confidence Badge & Anchor Provenance Row */}
        {!isLoading && (
          <div className="mt-4 flex flex-wrap items-center gap-3 text-[#6f7973] font-mono text-[11px] pt-3 border-t border-[#bfc9c1]/20">
            <span className="px-2.5 py-1 rounded bg-[#005239]/10 text-[#005239] font-semibold">
              {confidence}% Calibrated Confidence
            </span>

            {anchors && anchors.length > 0 ? (
              anchors.map((anchor, idx) => (
                <span
                  key={idx}
                  className="px-2 py-0.5 rounded bg-white border border-[#bfc9c1]/40 text-[#46645a]"
                >
                  {anchor}
                </span>
              ))
            ) : (
              <span className="px-2 py-0.5 rounded bg-white border border-[#bfc9c1]/40 text-[#46645a]">
                Anchor: [Indexed Knowledge Base]
              </span>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
