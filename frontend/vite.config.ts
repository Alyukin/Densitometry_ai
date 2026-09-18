import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// In dev mode requests to the API are proxied to the backend (default http://localhost:8000).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const target = env.VITE_DEV_API_PROXY || "http://localhost:8000";
  const proxy = Object.fromEntries(
    ["/api", "/health", "/docs", "/redoc", "/openapi.json"].map((p) => [p, { target, changeOrigin: true }]),
  );
  return {
    plugins: [react()],
    server: { host: true, port: 5173, proxy },
    preview: { port: 4173, proxy },
  };
});
