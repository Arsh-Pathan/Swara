#!/usr/bin/env bash
# Container and service health check script
set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-8000}"

URL="http://${HOST}:${PORT}/ready"

echo "Checking Swara health at $URL..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$URL" || echo "000")

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "Swara is HEALTHY (HTTP 200)"
    exit 0
else
    echo "Swara is UNHEALTHY (HTTP $HTTP_STATUS)"
    exit 1
fi
