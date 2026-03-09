#!/bin/bash
export PATH="/opt/homebrew/opt/node/bin:/opt/homebrew/bin:$PATH"
export NODE_PATH="/opt/homebrew/lib/node_modules"
export NEXT_PUBLIC_API_URL="http://localhost:8000"
cd "$(dirname "$0")"
exec /opt/homebrew/opt/node/bin/node /opt/homebrew/lib/node_modules/npm/bin/npm-cli.js run dev
