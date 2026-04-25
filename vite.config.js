import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/register': 'http://localhost:5000',
      '/login': 'http://localhost:5000',
      '/logout': 'http://localhost:5000',
      '/profile': 'http://localhost:5000',
      '/preferences': 'http://localhost:5000',
      '/summarize': 'http://localhost:5000',
      '/analyze': 'http://localhost:5000',
      '/extract': 'http://localhost:5000',
      '/keywords': 'http://localhost:5000',
      '/sentiment': 'http://localhost:5000',
    }
  }
})
