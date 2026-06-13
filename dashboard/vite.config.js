import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backendUrl = process.env.VITE_BACKEND_URL || (process.env.DOCKER_DEV ? 'http://backend:58001' : 'http://localhost:58001')
const rendererUrl = process.env.VITE_RENDERER_URL || (process.env.DOCKER_DEV ? 'http://renderer:3100' : 'http://localhost:3100')

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
        target: backendUrl,
        changeOrigin: true,
      },
      '/videos': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/thumbnails': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/gallery': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/video': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/render': {
        target: rendererUrl,
        changeOrigin: true,
      }
    }
  }
})
