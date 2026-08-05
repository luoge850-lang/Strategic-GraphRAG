/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
        mono: ['"Inter"', 'ui-monospace', 'monospace'],
      },
      colors: {
        paper: '#F0EFEB',
        ink: '#1C1C1A',
        muted: '#8F8E88',
        faint: '#C6C5BF',
        grid: '#DEDDD6',
      },
    },
  },
  plugins: [],
}
