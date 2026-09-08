import React from 'react';

const PROMPTS = [
  {
    icon: 'work_history',
    title: 'Experience & Key Projects',
    desc: 'Synthesize verified roles, team leadership, and major delivered software.',
    prompt: 'Summarize professional experience, chronological roles, and marquee enterprise projects',
    iconColor: 'text-[#005239]',
  },
  {
    icon: 'terminal',
    title: 'Technical Stack & Tools',
    desc: 'Enumerate languages, frameworks, cloud architecture, and toolsets.',
    prompt: 'Extract core technical skills, programming frameworks, and cloud infrastructure tools',
    iconColor: 'text-[#005239]',
  },
  {
    icon: 'school',
    title: 'Education & Credentials',
    desc: 'Academic credentials, university honors, and accredited certifications.',
    prompt: 'Analyze educational background, formal degrees, and industry certifications',
    iconColor: 'text-[#46645a]',
  },
  {
    icon: 'military_tech',
    title: 'Recognitions & Honors',
    desc: 'Patents, industry distinctions, hackathons, and published artifacts.',
    prompt: 'What are the highlighted achievements, publications, and competitive honors?',
    iconColor: 'text-[#46645a]',
  },
];

export default function SuggestedPrompts({ onSelectPrompt }) {
  return (
    <div className="w-full max-w-4xl mx-auto mt-6">
      <div className="flex items-center justify-between mb-2">
        <span className="font-['Manrope',sans-serif] text-[11px] font-semibold text-[#6f7973] uppercase tracking-wider">
          Suggested Exploration Prompts
        </span>
        <span className="font-mono text-[11px] text-[#6f7973]">
          Click to populate
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {PROMPTS.map((item, idx) => (
          <button
            key={idx}
            onClick={() => onSelectPrompt(item.prompt)}
            className="text-left flex items-start gap-3 p-3.5 rounded-lg bg-white hover:bg-[#f5f3f5] shadow-sm transition-all group border border-[#bfc9c1]/20 cursor-pointer"
            type="button"
          >
            <span className={`material-symbols-outlined ${item.iconColor} text-[20px] mt-0.5 flex-shrink-0 group-hover:scale-110 transition-transform`}>
              {item.icon}
            </span>
            <div className="min-w-0">
              <div className="font-semibold text-xs text-[#1b1c1e] group-hover:text-[#005239] transition-colors truncate">
                {item.title}
              </div>
              <p className="text-[11px] text-[#3f4943] line-clamp-1 mt-0.5">
                {item.desc}
              </p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
