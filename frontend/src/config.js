// Centralized API configuration supporting both local and decoupled (Vercel + Cloud) deployments
let rawBase = import.meta.env.VITE_API_BASE_URL || '';
rawBase = rawBase.replace(/\/+$/, '');

// If pointing to Hugging Face Spaces without /backend mount prefix, append /backend
// because Gradio 6 reserves root /api for internal CSRF-protected SvelteKit endpoints.
if (rawBase.includes('.hf.space') && !rawBase.endsWith('/backend')) {
  rawBase += '/backend';
}

export const API_BASE_URL = rawBase;
