import React, { useState, useEffect } from 'react';

export default function Hero() {
  const [greeting, setGreeting] = useState('Good evening');

  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting('Good morning');
    else if (hour < 17) setGreeting('Good afternoon');
    else setGreeting('Good evening');
  }, []);

  return (
    <section className="flex flex-col items-center text-center max-w-4xl mx-auto mb-4">
      <div className="w-full flex flex-col items-center gap-2 mb-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded border border-[#bfc9c1]/30 bg-[#f5f3f5]/70 font-mono text-[11px] uppercase tracking-widest text-[#46645a]">
          <span className="w-1.5 h-1.5 rounded-full bg-[#005239]"></span>
          <span>Archival Research Engine</span>
        </div>

        <div className="flex items-center justify-center gap-6 w-full mt-1">
          <div className="hidden md:block flex-1 h-[1px] bg-gradient-to-r from-transparent via-[#bfc9c1]/40 to-[#bfc9c1]/70"></div>
          <div className="flex flex-col items-center px-4">
            <div className="font-['EB_Garamond',serif] text-[48px] md:text-[56px] tracking-[0.28em] uppercase text-[#1b1c1e] font-normal leading-none pl-[0.28em]">
              V I O R A
            </div>
          </div>
          <div className="hidden md:block flex-1 h-[1px] bg-gradient-to-l from-transparent via-[#bfc9c1]/40 to-[#bfc9c1]/70"></div>
        </div>
      </div>

      <h1 className="font-['EB_Garamond',serif] text-3xl md:text-4xl text-[#1b1c1e] tracking-tight leading-tight mb-2 font-medium">
        {greeting}. What would you like to explore today?
      </h1>

      <p className="font-sans text-[0.9375rem] text-[#3f4943] max-w-2xl text-center leading-relaxed">
        Interrogate catalogued primary source documents with continuous evidentiary citation and deterministic ground-truth verification.
      </p>
    </section>
  );
}
