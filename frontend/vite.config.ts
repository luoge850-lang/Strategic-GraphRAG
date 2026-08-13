import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backend = { target: 'http://localhost:8000', changeOrigin: true }

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom'],
          motion: ['motion'],
          visData: ['vis-data'],
          visNetwork: ['vis-network'],
          icons: ['@phosphor-icons/react'],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: { '/api': { ...backend, rewrite: p => p.replace(/^\/api/, '') } },
  },
  // A production bundle normally shares the FastAPI origin and therefore
  // calls root routes such as /query.  Vite preview runs on its own origin;
  // proxy those same routes so production-build smoke tests exercise the
  // real backend instead of parsing index.html as JSON.
  preview: {
    port: 4173,
    proxy: {
      '/query': backend,
      '/graph': backend,
      '/evidence': backend,
      '/health': backend,
    },
  },
})
