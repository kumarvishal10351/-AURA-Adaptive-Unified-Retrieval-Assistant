import React from 'react';
import { BookOpen, FileText, Layers, ListChecks } from 'lucide-react';

const PROMPTS = [
  {
    icon: FileText,
    label: 'Executive summary',
    desc: 'Synthesize the main takeaways, purpose, and key conclusions',
    prompt: 'Provide an executive summary of the document, highlighting the main purpose and core takeaways.',
  },
  {
    icon: Layers,
    label: 'Key topics & architecture',
    desc: 'Extract key themes, frameworks, systems, and definitions',
    prompt: 'What are the main topics, technical systems, or frameworks discussed in the document?',
  },
  {
    icon: BookOpen,
    label: 'Findings & evidence',
    desc: 'Analyze specific data points, methodologies, and findings',
    prompt: 'What key findings, methodologies, and supporting evidence are presented in the document?',
  },
  {
    icon: ListChecks,
    label: 'Action items & conclusions',
    desc: 'Overview of recommendations, next steps, and insights',
    prompt: 'What conclusions, recommendations, and actionable insights are outlined in the document?',
  },
];

export default function SuggestedPrompts({ onSelectPrompt }) {
  return (
    <div className="w-full max-w-2xl mx-auto my-4">
      <div className="text-[11px] font-medium text-[#8b938e] uppercase tracking-wider mb-2 text-center sm:text-left">
        Suggested questions
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
        {PROMPTS.map((item, idx) => {
          const Icon = item.icon;
          return (
            <button
              key={idx}
              onClick={() => onSelectPrompt(item.prompt)}
              className="group text-left p-3.5 rounded-lg bg-white border border-[#e8e6e1] hover:border-[#124332]/40 hover:shadow-sm transition-all duration-150 cursor-pointer flex items-start gap-3"
              type="button"
            >
              <div className="p-1.5 rounded-md bg-[#f5f4f0] text-[#124332] group-hover:bg-[#124332] group-hover:text-white transition-colors flex-shrink-0 mt-0.5">
                <Icon className="w-4 h-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-xs font-semibold text-[#191b1a] group-hover:text-[#124332] transition-colors">
                  {item.label}
                </div>
                <div className="text-[11px] text-[#5e6661] mt-0.5 line-clamp-1">
                  {item.desc}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
