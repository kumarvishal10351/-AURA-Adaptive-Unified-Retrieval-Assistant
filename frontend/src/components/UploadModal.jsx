import React, { useState, useRef } from 'react';
import { Check, FileText, Loader2, Trash2, UploadCloud, X } from 'lucide-react';
import { API_BASE_URL } from '../config';

export default function UploadModal({
  isOpen,
  onClose,
  documents = [],
  onUploadSuccess,
  onDeleteDocument,
  onClearAll
}) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null); // { type: 'success'|'error', text: '' }
  const fileInputRef = useRef(null);

  if (!isOpen) return null;

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setUploadStatus({ type: 'info', text: `Uploading and indexing ${file.name}...` });

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setUploadStatus({
          type: 'success',
          text: `Successfully indexed ${file.name} (${data.chunks_count || 'all'} chunks)!`,
        });
        setTimeout(() => {
          setIsUploading(false);
          setUploadStatus(null);
          if (onUploadSuccess) onUploadSuccess();
        }, 1200);
      } else {
        setUploadStatus({
          type: 'error',
          text: `Upload failed: server responded with ${res.status}`,
        });
        setIsUploading(false);
      }
    } catch {
      setUploadStatus({
        type: 'error',
        text: 'Connection error while uploading document.',
      });
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/30 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-xl border border-[#e8e6e1] animate-slide-up">
        {/* Modal Header */}
        <div className="flex items-center justify-between pb-3 border-b border-[#f0eee9]">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md bg-[#eaf3ee] text-[#124332] flex items-center justify-center">
              <FileText className="w-4 h-4" />
            </div>
            <h2 className="text-base font-semibold text-[#191b1a]">Document Library</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-[#8b938e] hover:text-[#191b1a] hover:bg-[#f5f4f0] transition-colors cursor-pointer"
            type="button"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Upload Zone */}
        <div className="mt-4">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".pdf"
            className="hidden"
          />

          <div
            onClick={() => !isUploading && fileInputRef.current && fileInputRef.current.click()}
            className={`border border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
              isUploading
                ? 'border-[#124332] bg-[#f5fbf8]'
                : 'border-[#dcd9d2] hover:border-[#124332] hover:bg-[#faf9f6]'
            }`}
          >
            {isUploading ? (
              <div className="flex flex-col items-center justify-center gap-2 text-[#124332]">
                <Loader2 className="w-6 h-6 animate-spin" />
                <span className="text-xs font-medium">{uploadStatus?.text}</span>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center gap-1.5">
                <div className="w-10 h-10 rounded-full bg-[#f0eee9] text-[#124332] flex items-center justify-center mb-1">
                  <UploadCloud className="w-5 h-5" />
                </div>
                <div className="text-xs font-semibold text-[#191b1a]">
                  Click or drag PDF to index
                </div>
                <div className="text-[11px] text-[#8b938e]">
                  Supports PDF documents up to 50MB
                </div>
              </div>
            )}
          </div>

          {uploadStatus && !isUploading && (
            <div
              className={`mt-2.5 text-xs text-center py-1.5 px-2 rounded-md ${
                uploadStatus.type === 'success'
                  ? 'bg-[#eaf3ee] text-[#124332]'
                  : 'bg-red-50 text-red-700'
              }`}
            >
              {uploadStatus.text}
            </div>
          )}
        </div>

        {/* Currently Indexed Documents List */}
        <div className="mt-5">
          <div className="flex items-center justify-between mb-2">
            <div className="text-[11px] font-medium text-[#8b938e] uppercase tracking-wider">
              Active Documents ({documents.length})
            </div>
            {documents.length > 0 && onClearAll && (
              <button
                onClick={onClearAll}
                className="text-[11px] text-red-600 hover:text-red-700 hover:underline cursor-pointer"
                type="button"
              >
                Clear all
              </button>
            )}
          </div>

          <div className="max-h-52 overflow-y-auto space-y-1.5 pr-1">
            {documents.length > 0 ? (
              documents.map((doc, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-[#faf9f6] border border-[#f0eee9] text-xs text-[#191b1a]"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <FileText className="w-3.5 h-3.5 text-[#124332] flex-shrink-0" />
                    <span className="truncate font-medium">{doc.name}</span>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                    {doc.size && (
                      <span className="text-[10px] font-mono text-[#8b938e]">{doc.size}</span>
                    )}
                    <span className="inline-flex items-center text-[10px] text-[#124332] bg-[#eaf3ee] px-1.5 py-0.5 rounded font-medium">
                      <Check className="w-2.5 h-2.5 mr-0.5" /> Indexed
                    </span>
                    {onDeleteDocument && (
                      <button
                        onClick={() => onDeleteDocument(doc.name)}
                        className="p-1 text-[#8b938e] hover:text-red-600 rounded transition-colors cursor-pointer"
                        title={`Remove ${doc.name}`}
                        type="button"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-xs text-[#8b938e] text-center py-5 border border-dashed border-[#e8e6e1] rounded-lg">
                No documents indexed yet. Upload a PDF above to get started.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
