import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: [
      'openshorts.app',
      'www.openshorts.app'
    ],
    proxy: {
      '/api': {
        target: process.env.VITE_BACKEND_URL || 'http://localhost:58001',
        changeOrigin: true,
      },
      '/videos': {
        target: process.env.VITE_BACKEND_URL || 'http://localhost:58001',
        changeOrigin: true,
      },
      '/thumbnails': {
        target: process.env.VITE_BACKEND_URL || 'http://localhost:58001',
        changeOrigin: true,
      },
      '/gallery': {
        target: process.env.VITE_BACKEND_URL || 'http://localhost:58001',
        changeOrigin: true,
      },
      '/video': {
        target: process.env.VITE_BACKEND_URL || 'http://localhost:58001',
        changeOrigin: true,
      },
      '/render': {
        target: process.env.VITE_RENDERER_URL || 'http://localhost:3100',
        changeOrigin: true,
      }
    }
  }
})
