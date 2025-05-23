#!/usr/bin/env bash
# test_image_search_agent.sh
# ------------------------------------------------------------
# Sends an image‐search query to the agent-service’s 
# /search_images endpoint and pretty‐prints the list of URLs.
# ------------------------------------------------------------
set -euo pipefail

# ---------- Config ----------
QUERY=${1:-"Intel AI PCs"}
ENDPOINT=${2:-http://127.0.0.1:8001/search_images}

# ---------- Wait for service to be healthy ----------
printf "⏳ Waiting for agent-service to be up"
for i in {1..20}; do
  if curl -fs "${ENDPOINT/\/search_images/\/health}" >/dev/null; then
    break
  fi
  printf "."
  sleep 0.5
done
echo

# ---------- Perform image search + display results ----------
echo "📤  Searching images for: \"$QUERY\" → $ENDPOINT"
curl -s -X POST \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"$QUERY\"}" \
     "$ENDPOINT" \
  | jq .

echo
echo "✅  Done."