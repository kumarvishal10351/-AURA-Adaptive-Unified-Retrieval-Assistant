import React, { useRef, useEffect, useState } from 'react';
import { ArrowUp, Check, ChevronDown, FileText, Files, Filter, Loader2, Plus, Sparkles, Target } from 'lucide-react';

export default function QueryComposer({
  query,
  setQuery,
  onSend,
  selectedDoc = 'all',
  setSelectedDoc,
  documents = [],
  onOpenUpload,
  isLoading,
}) {
  const textareaRef = useRef(null);
  const scrollContainerRef = useRef(null);
  const dropdownRef = useRef(null);
  const [isScopeMenuOpen, setIsScopeMenuOpen] = useState(false);

  // Close scope dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsScopeMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [query]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && query.trim()) {
        onSend();
      }
    }
  };

  const handleSelectDoc = (docName) => {
    if (setSelectedDoc) {
      setSelectedDoc(docName);
    }
    setIsScopeMenuOpen(false);
  };

  return (
    <div className="w-full max-w-3xl mx-auto relative">
      {/* Target / Scope Selection Popover Menu */}
      {isScopeMenuOpen && (
        <div
          ref={dropdownRef}
          className="absolute bottom-full mb-2 left-0 w-80 max-w-[90vw] bg-white/95 backdrop-blur-md rounded-xl border border-[#e8e6e1] shadow-lg p-2 z-50 animate-slide-up"
        >
          <div className="flex items-center justify-between px-2.5 py-1.5 border-b border-[#f0eee9] mb-1">
            <span className="text-[11px] font-semibold text-[#191b1a] uppercase tracking-wider flex items-center gap-1.5">
              <Target className="w-3.5 h-3.5 text-[#124332]" />
              Select Target Scope
            </span>
            <span className="text-[10px] text-[#8b938e] font-mono">
              {documents.length} available
            </span>
          </div>

          <div className="max-h-56 overflow-y-auto space-y-1 py-1 pr-1">
            {/* All Documents Option */}
            <button
              onClick={() => handleSelectDoc('all')}
              className={`w-full flex items-center justify-between px-2.5 py-2 rounded-lg text-xs font-medium transition-all text-left cursor-pointer ${
                selectedDoc === 'all'
                  ? 'bg-[#124332] text-white shadow-2xs'
                  : 'hover:bg-[#f5f4f0] text-[#191b1a]'
              }`}
              type="button"
            >
              <div className="flex items-center gap-2">
                <Files className="w-3.5 h-3.5 flex-shrink-0" />
                <span>All Documents ({documents.length})</span>
              </div>
              {selectedDoc === 'all' && <Check className="w-3.5 h-3.5 flex-shrink-0" />}
            </button>

            {/* Individual Documents */}
            {documents.map((doc, idx) => {
              const isSelected = selectedDoc === doc.name;
              return (
                <button
                  key={idx}
                  onClick={() => handleSelectDoc(doc.name)}
                  className={`w-full flex items-center justify-between px-2.5 py-2 rounded-lg text-xs font-medium transition-all text-left cursor-pointer ${
                    isSelected
                      ? 'bg-[#124332] text-white shadow-2xs'
                      : 'hover:bg-[#f5f4f0] text-[#191b1a]'
                  }`}
                  title={doc.name}
                  type="button"
                >
                  <div className="flex items-center gap-2 min-w-0 pr-2">
                    <FileText className="w-3.5 h-3.5 flex-shrink-0 opacity-80" />
                    <span className="truncate">{doc.name}</span>
                  </div>
                  <div className="flex items-center gap-1.5 flex-shrink-0">
                    {doc.size && (
                      <span className={`text-[10px] font-mono ${isSelected ? 'text-white/80' : 'text-[#8b938e]'}`}>
                        {doc.size}
                      </span>
                    )}
                    {isSelected && <Check className="w-3.5 h-3.5" />}
                  </div>
                </button>
              );
            })}
          </div>

          {documents.length === 0 && (
            <div className="p-3 text-center text-xs text-[#8b938e]">
              No documents indexed. Upload a PDF first.
            </div>
          )}
        </div>
      )}

      {/* Main Composer Box with Translucent Glassmorphism */}
      <div className="relative bg-white/45 backdrop-blur-sm rounded-2xl border border-[#e8e6e1]/60 shadow-sm hover:border-[#124332]/30 focus-within:border-[#124332] focus-within:ring-1 focus-within:ring-[#124332]/20 transition-all p-2.5 sm:p-3 flex flex-col gap-2">
        {/* Shimmer loading progress bar across top */}
        {isLoading && (
          <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[#124332] to-transparent animate-pulse" />
        )}

        {/* Dedicated Document Scope Selection Header Row */}
        <div className="flex items-center gap-2 pb-2 border-b border-[#f0eee9]/80">
          {/* Dedicated Target / Scope Button */}
          <button
            onClick={() => setIsScopeMenuOpen(!isScopeMenuOpen)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-[#124332] text-white hover:bg-[#195842] transition-all cursor-pointer shadow-2xs flex-shrink-0"
            type="button"
            title="Click to select target document scope"
          >
            <Target className="w-3.5 h-3.5" />
            <span className="max-w-[130px] sm:max-w-[170px] truncate">
              {selectedDoc === 'all' ? `Scope: All (${documents.length})` : `Target: ${selectedDoc}`}
            </span>
            <ChevronDown className={`w-3 h-3 transition-transform ${isScopeMenuOpen ? 'rotate-180' : ''}`} />
          </button>

          {documents.length > 0 ? (
            /* Horizontal Slideable Container for Quick Pills */
            <div
              ref={scrollContainerRef}
              className="flex-1 flex items-center gap-1.5 overflow-x-auto no-scrollbar scroll-smooth py-0.5"
            >
              {/* All Documents Quick Pill */}
              <button
                onClick={() => handleSelectDoc('all')}
                className={`flex-shrink-0 flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium transition-all cursor-pointer ${
                  selectedDoc === 'all'
                    ? 'bg-[#eaf3ee] text-[#124332] border border-[#124332]/30 font-semibold'
                    : 'bg-[#faf9f6] text-[#5e6661] border border-[#e8e6e1] hover:border-[#124332]/40 hover:text-[#191b1a]'
                }`}
                type="button"
              >
                <Files className="w-3 h-3" />
                <span>All ({documents.length})</span>
              </button>

              {/* Individual Document Quick Pills */}
              {documents.map((doc, idx) => {
                const isSelected = selectedDoc === doc.name;
                return (
                  <button
                    key={idx}
                    onClick={() => handleSelectDoc(doc.name)}
                    className={`flex-shrink-0 flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium transition-all cursor-pointer max-w-[180px] ${
                      isSelected
                        ? 'bg-[#eaf3ee] text-[#124332] border border-[#124332]/30 font-semibold'
                        : 'bg-[#faf9f6] text-[#5e6661] border border-[#e8e6e1] hover:border-[#124332]/40 hover:text-[#191b1a]'
                    }`}
                    title={doc.name}
                    type="button"
                  >
                    <FileText className="w-3 h-3 flex-shrink-0" />
                    <span className="truncate">{doc.name}</span>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-between">
              <button
                onClick={onOpenUpload}
                className="flex items-center gap-1 text-xs font-medium text-[#8b938e] hover:text-[#124332] transition-colors cursor-pointer truncate"
                type="button"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>No documents indexed yet — click to add PDF</span>
              </button>
            </div>
          )}
        </div>

        {/* Input Textarea */}
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            selectedDoc === 'all'
              ? 'Ask a question across all documents...'
              : `Ask a question about ${selectedDoc}...`
          }
          rows={1}
          disabled={isLoading}
          className="w-full resize-none border-none outline-none bg-transparent text-sm text-[#191b1a] placeholder:text-[#8b938e] leading-relaxed max-h-40 px-1 py-1"
        />

        {/* Action Bottom Row */}
        <div className="flex items-center justify-end pt-1 border-t border-[#f0eee9]/80">
          <button
            onClick={onSend}
            disabled={isLoading || !query.trim()}
            className="w-8 h-8 rounded-full bg-[#124332] text-white flex items-center justify-center hover:bg-[#195842] disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-xs cursor-pointer"
            type="button"
            title="Send message"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin text-white" />
            ) : (
              <ArrowUp className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
