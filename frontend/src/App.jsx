import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import SuggestedPrompts from './components/SuggestedPrompts';
import ChatStream from './components/ChatStream';
import QueryComposer from './components/QueryComposer';
import UploadModal from './components/UploadModal';
import { API_BASE_URL } from './config';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [docCount, setDocCount] = useState(0);
  const [selectedDoc, setSelectedDoc] = useState('all');
  const [fallbackLoadingId, setFallbackLoadingId] = useState(null);

  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const chatBottomRef = useRef(null);

  // Fetch status and document list
  const fetchStatusAndDocs = async () => {
    try {
      const statusRes = await fetch(`${API_BASE_URL}/api/status`);
      if (statusRes.ok) {
        const data = await statusRes.json();
        if (data.total_docs !== undefined) {
          setDocCount(data.total_docs);
        }
      }

      const docRes = await fetch(`${API_BASE_URL}/api/documents`);
      if (docRes.ok) {
        const data = await docRes.json();
        if (data.documents) {
          setDocuments(data.documents);
          setDocCount(data.documents.length);
          // If selected doc was removed, reset to 'all'
          if (selectedDoc !== 'all' && !data.documents.some((d) => d.name === selectedDoc)) {
            setSelectedDoc('all');
          }
        }
      }
    } catch {
      // Backend starting up or standalone mode
    }
  };

  // Load active documents and status on initial mount (never wipe database on reload)
  useEffect(() => {
    fetchStatusAndDocs();
  }, []);

  // Smooth scroll to bottom when messages update or loading
  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  const handleSendMessage = async (customPrompt) => {
    const q = (customPrompt || query).trim();
    if (!q || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    // Prepare multi-turn history for backend
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setQuery('');
    setIsLoading(true);

    const startT = performance.now();
    const historyPayload = nextMessages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      const res = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          history: historyPayload,
          selected_doc: selectedDoc,
        }),
      });

      const elapsed = Math.round(performance.now() - startT);

      if (res.ok) {
        const data = await res.json();
        const assistantMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.answer || 'No grounded answer returned for this question.',
          sources: data.sources || [],
          latency: data.latency_ms || elapsed,
          can_fallback: data.can_fallback || false,
          is_fallback: data.is_fallback || false,
          userQuery: q,
          isNew: true,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        throw new Error(`Server returned error ${res.status}`);
      }
    } catch {
      const elapsed = Math.round(performance.now() - startT);
      const fallbackMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Unable to connect to the assistant service at this moment. Please check your connection and try again, or consult the general-knowledge fallback model.',
        sources: [],
        latency: elapsed,
        can_fallback: true,
        is_fallback: false,
        userQuery: q,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, fallbackMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Dedicated Fallback trigger for an existing message
  const handleTriggerFallback = async (queryText, messageId) => {
    if (!queryText || fallbackLoadingId) return;
    setFallbackLoadingId(messageId);

    const startT = performance.now();
    try {
      const res = await fetch(`${API_BASE_URL}/api/fallback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText,
          history: messages.map((m) => ({ role: m.role, content: m.content })),
          selected_doc: selectedDoc,
        }),
      });

      const elapsed = Math.round(performance.now() - startT);

      if (res.ok) {
        const data = await res.json();
        const fallbackMsgId = `${messageId}-fallback-${Date.now()}`;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === messageId
              ? {
                  ...m,
                  id: fallbackMsgId,
                  content: data.answer,
                  sources: data.sources || [],
                  latency: data.latency_ms || elapsed,
                  can_fallback: false,
                  is_fallback: true,
                  isNew: true,
                }
              : m
          )
        );
      }
    } catch (err) {
      console.error('Fallback query error:', err);
    } finally {
      setFallbackLoadingId(null);
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setQuery('');
  };

  const handleSelectPrompt = (promptText) => {
    handleSendMessage(promptText);
  };

  const handleDeleteDocument = async (filename) => {
    try {
      await fetch(`${API_BASE_URL}/api/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      fetchStatusAndDocs();
    } catch {
      // Backend error handling
    }
  };

  const handleClearAllDocuments = async () => {
    try {
      await fetch(`${API_BASE_URL}/api/documents`, { method: 'DELETE' });
      fetchStatusAndDocs();
    } catch {
      // Backend error handling
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-between bg-[#fbfbfa] relative">
      {/* Background App Watermark with Luxurious Architectural Typography */}
      <div
        aria-hidden="true"
        className="fixed inset-0 flex items-center justify-center pointer-events-none select-none z-0 overflow-hidden"
      >
        <span className="font-['Cinzel',serif] font-black text-[18vw] md:text-[230px] lg:text-[290px] tracking-[0.28em] text-[#124332]/[0.085] uppercase select-none leading-none pl-[0.28em] transition-all">
          VIORA
        </span>
      </div>

      <div className="relative z-10 flex-1 flex flex-col justify-between">
        {/* Fixed Pinned Header */}
        <Header
          docCount={docCount}
          hasMessages={messages.length > 0}
          onOpenDocuments={() => setIsModalOpen(true)}
          onOpenUpload={() => setIsModalOpen(true)}
          onNewChat={handleNewChat}
        />

        {/* Main Content Viewport with Room for Fixed Header (pt-20) and Composer (pb-56) */}
        <main className="flex-1 flex flex-col w-full max-w-4xl mx-auto px-4 sm:px-6 pt-20 pb-56">
          {/* Empty State / Welcome Screen */}
          {messages.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center min-h-[50vh] gap-4">
              <Hero />
              <SuggestedPrompts onSelectPrompt={handleSelectPrompt} />
            </div>
          ) : (
            /* Active Chat Stream */
            <div className="flex-1">
              <ChatStream
                messages={messages}
                isLoading={isLoading}
                onTriggerFallback={handleTriggerFallback}
                fallbackLoadingId={fallbackLoadingId}
              />
              <div ref={chatBottomRef} />
            </div>
          )}
        </main>
      </div>

      {/* Docked Input Area with Translucent Glassmorphism */}
      <div className="fixed bottom-0 left-0 right-0 z-30 bg-gradient-to-t from-[#fbfbfa]/60 via-[#fbfbfa]/30 to-transparent backdrop-blur-[2px] pt-6 pb-3 px-4 sm:px-6">
        <QueryComposer
          query={query}
          setQuery={setQuery}
          onSend={() => handleSendMessage()}
          selectedDoc={selectedDoc}
          setSelectedDoc={setSelectedDoc}
          documents={documents}
          onOpenUpload={() => setIsModalOpen(true)}
          isLoading={isLoading}
        />
      </div>

      {/* Document Library / Upload Modal */}
      <UploadModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        documents={documents}
        onUploadSuccess={fetchStatusAndDocs}
        onDeleteDocument={handleDeleteDocument}
        onClearAll={handleClearAllDocuments}
      />
    </div>
  );
}
