import React, { useState, useEffect, useRef } from 'react';
import { BookOpen, Check, Copy, ExternalLink, FileText, Loader2, Sparkles } from 'lucide-react';

// Word-by-word typewriter renderer mimicking ChatGPT streaming
function TypewriterMessage({
  text = '',
  isNew = false,
  onComplete,
  renderFormattedText,
}) {
  const [displayedCount, setDisplayedCount] = useState(() => (isNew ? 0 : null));
  const tokensRef = useRef(text ? text.split(/(\s+)/) : []);

  useEffect(() => {
    if (!isNew) {
      setDisplayedCount(null);
      return;
    }

    const tokens = text.split(/(\s+)/);
    tokensRef.current = tokens;
    let count = 0;
    setDisplayedCount(0);

    // Dynamic token increment ensuring the typewriter finishes smoothly in ~1-1.2s max
    const step = Math.max(3, Math.ceil(tokens.length / 50));
    const timer = setInterval(() => {
      count += step;
      if (count >= tokens.length) {
        clearInterval(timer);
        setDisplayedCount(null);
        if (onComplete) onComplete();
      } else {
        setDisplayedCount(count);
      }
    }, 14);

    return () => clearInterval(timer);
  }, [text, isNew]);

  if (displayedCount === null) {
    return renderFormattedText(text);
  }

  const currentSnippet = tokensRef.current.slice(0, displayedCount).join('');
  return (
    <div className="relative">
      {renderFormattedText(currentSnippet)}
      <span className="inline-block w-1.5 h-4 ml-1 bg-[#124332] animate-pulse align-middle" />
    </div>
  );
}

export default function ChatStream({
  messages = [],
  isLoading = false,
  onTriggerFallback,
  fallbackLoadingId = null,
}) {
  const [copiedId, setCopiedId] = useState(null);
  const [expandedSources, setExpandedSources] = useState({});
  const [completedAnimationIds, setCompletedAnimationIds] = useState({});

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  const toggleSourceExpand = (messageId, sourceIdx) => {
    const key = `${messageId}-${sourceIdx}`;
    setExpandedSources((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const markAnimationComplete = (id) => {
    setCompletedAnimationIds((prev) => ({ ...prev, [id]: true }));
  };

  // Helper to format text with bullets and bolding cleanly
  const renderFormattedText = (text) => {
    if (!text) return null;

    const lines = text.split('\n');
    return (
      <div className="space-y-2 leading-relaxed text-[0.9375rem] text-[#191b1a] font-sans">
        {lines.map((line, i) => {
          const trimmed = line.trim();
          if (!trimmed) {
            return <div key={i} className="h-1.5" />;
          }

          // Bullet line
          if (trimmed.startsWith('•') || trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
            const content = trimmed.replace(/^[•\-*]\s*/, '');
            return (
              <div key={i} className="flex items-start gap-2 pl-1">
                <span className="text-[#124332] font-bold text-sm select-none">•</span>
                <span className="flex-1">{formatInline(content)}</span>
              </div>
            );
          }

          return <p key={i}>{formatInline(line)}</p>;
        })}
      </div>
    );
  };

  // Inline formatting helper for bold text (**text**)
  const formatInline = (str) => {
    const parts = str.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, idx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={idx} className="font-semibold text-[#191b1a]">
            {part.slice(2, -2)}
          </strong>
        );
      }
      return part;
    });
  };

  // Helper to retrieve the original user question for an assistant message
  const getQueryForMessage = (msg, msgIdx) => {
    if (msg.userQuery) return msg.userQuery;
    for (let i = msgIdx - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        return messages[i].content;
      }
    }
    return '';
  };

  return (
    <div className="w-full max-w-3xl mx-auto flex flex-col gap-6 py-4">
      {messages.map((msg, idx) => {
        const isUser = msg.role === 'user';

        if (isUser) {
          return (
            <div key={msg.id} className="flex justify-end animate-slide-up">
              {/* User Bubble with Translucent Glassmorphism */}
              <div className="max-w-[85%] sm:max-w-[75%] rounded-2xl px-4 py-3 bg-[#f0eee9]/40 backdrop-blur-sm text-[#191b1a] text-sm leading-relaxed border border-[#e8e6e1]/50 shadow-2xs">
                {msg.content}
              </div>
            </div>
          );
        }

        const userQuery = getQueryForMessage(msg, idx);
        const lowerContent = (msg.content || '').toLowerCase();
        const hasInsufficientEvidence =
          msg.can_fallback ||
          lowerContent.includes('not_found') ||
          lowerContent.includes('not found') ||
          lowerContent.includes('does not contain sufficient grounded evidence') ||
          lowerContent.includes('does not contain any information') ||
          lowerContent.includes('context does not contain') ||
          lowerContent.includes('cannot find any information') ||
          lowerContent.includes('no information') ||
          lowerContent.includes('do not define or explain') ||
          lowerContent.includes('not mentioned in the');
        const shouldAnimate = msg.isNew && !completedAnimationIds[msg.id];
        const isTypingDone = !shouldAnimate;
        const displayContent = (msg.content || '').replace(/^\*{0,2}NOT[ _]FOUND\*{0,2}[\s:\-\n]*/i, '');

        // Assistant Message
        return (
          <div key={msg.id} className="flex flex-col gap-3 animate-slide-up">
            {/* Assistant Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-md bg-[#124332] text-white flex items-center justify-center shadow-2xs">
                  <BookOpen className="w-3.5 h-3.5" />
                </div>
                <span className="text-xs font-semibold text-[#191b1a]">Viora</span>

                {/* Fallback Model Pill Badge */}
                {msg.is_fallback && (
                  <span className="ml-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium bg-[#eaf3ee] text-[#124332] border border-[#124332]/20 flex items-center gap-1 shadow-2xs">
                    <Sparkles className="w-3 h-3 text-[#124332]" />
                    <span>Fallback Model • Mistral Large</span>
                  </span>
                )}
              </div>

              {msg.latency && (
                <span className="text-[11px] font-mono text-[#8b938e]">
                  {msg.latency}ms
                </span>
              )}
            </div>

            {/* Answer Card with Translucent Glassmorphism */}
            <div className="bg-white/35 backdrop-blur-sm rounded-xl p-5 border border-[#e8e6e1]/60 shadow-2xs transition-all">
              <TypewriterMessage
                key={msg.id}
                text={displayContent}
                isNew={shouldAnimate}
                onComplete={() => markAnimationComplete(msg.id)}
                renderFormattedText={renderFormattedText}
              />

              {/* Fallback Callout: ONLY appears when Not-Found / Insufficient Evidence is Output */}
              {isTypingDone && hasInsufficientEvidence && !msg.is_fallback && (
                <div className="mt-4 p-3.5 rounded-lg bg-white/60 backdrop-blur-sm border border-[#e8e6e1]/70 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 animate-fade-in">
                  <div className="flex items-start sm:items-center gap-2.5">
                    <Sparkles className="w-4 h-4 text-[#124332] flex-shrink-0 mt-0.5 sm:mt-0" />
                    <div className="text-xs text-[#5e6661]">
                      <span className="font-semibold text-[#191b1a] block sm:inline">
                        Answer not in uploaded documents?
                      </span>{' '}
                      <span>Consult Viora's general-knowledge fallback model.</span>
                    </div>
                  </div>
                  <button
                    onClick={() => onTriggerFallback && onTriggerFallback(userQuery, msg.id)}
                    disabled={fallbackLoadingId === msg.id}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium text-white bg-[#124332] hover:bg-[#195842] transition-all cursor-pointer shadow-xs disabled:opacity-50 flex-shrink-0"
                    type="button"
                  >
                    {fallbackLoadingId === msg.id ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Consulting Fallback...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Consult Mistral Large (Fallback)</span>
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Citations & Sources Section (Revealed once streaming completes) */}
              {isTypingDone && msg.sources && msg.sources.length > 0 && (
                <div className="mt-5 pt-4 border-t border-[#f0eee9]/80 animate-fade-in">
                  <div className="text-[11px] font-medium text-[#8b938e] uppercase tracking-wider mb-2.5">
                    Sources Referenced ({msg.sources.length})
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {msg.sources.map((src, sIdx) => {
                      const key = `${msg.id}-${sIdx}`;
                      const isExpanded = !!expandedSources[key];
                      const fileName = src.file_name || 'Document';
                      const pageNum = src.page ? `p. ${src.page}` : '';

                      return (
                        <div key={sIdx} className="flex flex-col gap-1.5">
                          <button
                            onClick={() => toggleSourceExpand(msg.id, sIdx)}
                            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
                              isExpanded
                                ? 'bg-[#eaf3ee] border-[#124332] text-[#124332]'
                                : 'bg-[#faf9f6]/90 border-[#e8e6e1] text-[#5e6661] hover:border-[#124332]/40 hover:text-[#191b1a]'
                            }`}
                            type="button"
                          >
                            <FileText className="w-3 h-3 text-[#124332]" />
                            <span className="truncate max-w-[180px]">{fileName}</span>
                            {pageNum && (
                              <span className="text-[10px] opacity-75 font-mono">({pageNum})</span>
                            )}
                            <ExternalLink className="w-2.5 h-2.5 opacity-60 ml-0.5" />
                          </button>

                          {/* Excerpt Accordion */}
                          {isExpanded && src.preview && (
                            <div className="p-3 bg-[#faf9f6]/95 backdrop-blur-sm rounded-md border border-[#e8e6e1] text-xs text-[#5e6661] leading-relaxed max-w-md animate-slide-up">
                              <div className="font-medium text-[#191b1a] text-[11px] mb-1 flex items-center justify-between">
                                <span>Excerpt from {fileName}</span>
                                {src.score !== undefined && (
                                  <span className="font-mono text-[10px] text-[#124332]">
                                    Match: {Math.round(src.score * 100)}%
                                  </span>
                                )}
                              </div>
                              <p className="italic">"{src.preview.trim()}"</p>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Action Toolbar */}
              {isTypingDone && (
                <div className="mt-4 pt-3 flex items-center justify-between border-t border-[#f0eee9]/80 animate-fade-in">
                  <button
                    onClick={() => handleCopy(msg.id, msg.content)}
                    className="flex items-center gap-1.5 text-xs text-[#5e6661] hover:text-[#191b1a] transition-colors cursor-pointer py-1 px-1.5 rounded hover:bg-[#faf9f6]"
                    type="button"
                  >
                    {copiedId === msg.id ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-[#124332]" />
                        <span className="text-[#124332] font-medium">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>Copy answer</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>
        );
      })}

      {/* Thinking / In-Flight Indicator with Translucent Glassmorphism */}
      {isLoading && (
        <div className="flex flex-col gap-2.5 animate-slide-up">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-[#124332] text-white flex items-center justify-center">
              <BookOpen className="w-3.5 h-3.5" />
            </div>
            <span className="text-xs font-semibold text-[#191b1a]">Viora</span>
          </div>

          <div className="relative overflow-hidden bg-white/35 backdrop-blur-sm rounded-xl p-5 border border-[#e8e6e1]/60 shadow-2xs flex flex-col gap-3">
            {/* Shimmer top accent */}
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[#124332] to-transparent animate-pulse" />

            <div className="flex items-center gap-2 text-xs font-medium text-[#124332]">
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#124332] opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#124332]"></span>
              </span>
              <span>Reviewing documents & synthesizing response...</span>
            </div>

            {/* Skeleton placeholder bars with shimmer */}
            <div className="space-y-2 pt-1">
              <div className="h-3.5 bg-[#f0eee9]/80 rounded-md w-[85%] animate-pulse" />
              <div className="h-3.5 bg-[#f0eee9]/80 rounded-md w-[70%] animate-pulse" style={{ animationDelay: '0.15s' }} />
              <div className="h-3.5 bg-[#f0eee9]/80 rounded-md w-[50%] animate-pulse" style={{ animationDelay: '0.3s' }} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
