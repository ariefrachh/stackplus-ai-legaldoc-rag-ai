/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],

  theme: {
    extend: {
      colors: {
        primary: "#1D4ED8",     // Biru utama StackPlus
        secondary: "#EFF6FF",   // Background biru muda
        dark: "#0F172A",        // Text heading
        muted: "#64748B",       // Text sekunder
        border: "#E2E8F0",      // Border clean
        success: "#16A34A",
        danger: "#DC2626",
      },

      boxShadow: {
        soft: "0 10px 30px rgba(0,0,0,0.06)",
        card: "0 4px 20px rgba(0,0,0,0.05)",
      },

      borderRadius: {
        xl2: "1rem",
        xl3: "1.25rem",
      },

      fontFamily: {
        sans: ["Inter", "sans-serif"],
      },

      maxWidth: {
        chat: "720px",
      }
    },
  },

  plugins: [],
};