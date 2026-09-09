import React from 'react';
import { BookOpen, FileText, Plus, RotateCcw } from 'lucide-react';

export default function Header({
  docCount = 0,
  hasMessages = false,
  onOpenDocuments,
  onOpenUpload,
  onNewChat
}) {
  return (
    <header className="fixed top-0 left-0 right-0 z-40 w-full bg-[#ffffff]/90 backdrop-blur-md border-b border-[#e8e6e1]">
      <div className="max-w-5xl mx-auto h-14 px-4 sm:px-6 flex items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#124332] text-white flex items-center justify-center shadow-sm">
            <BookOpen className="w-4 h-4" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-sans text-xl font-semibold tracking-tight text-[#191b1a]">
              Viora
            </span>
            <span className="text-[11px] font-medium text-[#5e6661] hidden sm:inline-block">
              Research Assistant
            </span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {/* Document Indicator Button */}
          <button
            onClick={onOpenDocuments}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#5e6661] hover:text-[#191b1a] bg-[#f5f4f0] hover:bg-[#eceae4] rounded-md transition-colors cursor-pointer"
            title="View indexed documents"
            type="button"
          >
            <FileText className="w-3.5 h-3.5 text-[#124332]" />
            <span>{docCount} {docCount === 1 ? 'Document' : 'Documents'}</span>
          </button>

          {/* New Chat Button */}
          {hasMessages && (
            <button
              onClick={onNewChat}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#5e6661] hover:text-[#191b1a] hover:bg-[#f5f4f0] rounded-md transition-colors cursor-pointer"
              title="Start a new conversation"
              type="button"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">New Chat</span>
            </button>
          )}

          {/* Upload Button */}
          <button
            onClick={onOpenUpload}
            className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium text-white bg-[#124332] hover:bg-[#195842] rounded-md transition-colors shadow-sm cursor-pointer"
            type="button"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Add PDF</span>
          </button>
        </div>
      </div>
    </header>
  );
}
