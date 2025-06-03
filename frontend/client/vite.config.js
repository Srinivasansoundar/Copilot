import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  server:{
    proxy:{
      '/api':{
        target:'http://127.0.0.1:5000',
        secure:false,
      },
    },
  },
  plugins: [react(),tailwindcss()],
  
})
