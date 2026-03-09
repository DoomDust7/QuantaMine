import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // "standalone" is for Docker; remove for Vercel (uses its own output)
  // output: "standalone",
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  },
};

export default nextConfig;
