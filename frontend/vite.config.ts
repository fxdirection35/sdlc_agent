import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Dev-mode proxy: the SPA talks to /api on its own origin, and Vite
    // forwards to the FastAPI service. In a built deployment there is no
    // proxy - the service itself serves the built files, so same-origin holds.
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
  },
});
