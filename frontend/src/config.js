// Centralized API configuration supporting both local and decoupled (Vercel + Cloud) deployments
const rawBase = import.meta.env.VITE_API_BASE_URL || '';
export const API_BASE_URL = rawBase.replace(/\/+$/, '');
