/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      './pages/**/*.{js,ts,jsx,tsx}',
      './src/**/*.{js,ts,jsx,tsx}',
    ],
    theme: {
      extend: {
        colors: {
          
          primary: '#3b82f6', // Blue
          secondary: '#9333ea', // Purple
          danger: '#ef4444', // Red
          success: '#10b981', // Green
          warning: '#f59e0b', // Yellow
          background: '#f9fafb', // Light background for the app
          darkBackground: '#1f2937', // Dark background for dark mode
          textPrimary: '#111827',
          textSecondary: '#6b7280',
        },
        fontFamily: {
          // Custom fonts
          sans: ['Inter', 'sans-serif'],
          serif: ['Merriweather', 'serif'],
        },
        spacing: {
          // Custom spacing scale
          128: '32rem', // Example of a custom spacing unit
        },
        screens: {
          // Custom screen breakpoints
          'xs': '475px', // Extra small devices
        },
      },
    },
    plugins: [
      // Adding Tailwind's plugin for forms if needed
      require('@tailwindcss/forms'),
      // Adding Tailwind's plugin for aspect-ratio if needed
      require('@tailwindcss/aspect-ratio'),
      // Adding Tailwind's plugin for typography (prose) if needed
      require('@tailwindcss/typography'),
    ],
  }
   