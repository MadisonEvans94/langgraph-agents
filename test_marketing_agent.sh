#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
PDF_PATH="sample_input.pdf"
ENDPOINT="${1:-http://127.0.0.1:8001/run_marketing_agent}"

# ---------- Sanity checks ----------
if [[ ! -f "$PDF_PATH" ]]; then
  echo "❌  PDF not found: $PDF_PATH" >&2
  exit 1
fi

# ---------- Wait for service ----------
printf "⏳ Waiting for agent-service"
for i in {1..20}; do
  if curl -s -o /dev/null "$ENDPOINT"; then break; fi
  printf "."
  sleep 0.5
done
echo

# ---------- Invoke supervisor agent ----------
echo "📤  Uploading $PDF_PATH → $ENDPOINT"
response=$(curl -s -X POST \
  -F "file=@${PDF_PATH};type=application/pdf" \
  "$ENDPOINT")

# ---------- Pretty‐print results ----------
echo
echo "📝 Summary:"
echo "$response" | jq -r '.summary'
echo
echo "🔑 Key Points:"
echo "$response" | jq -r '.key_points[]'
echo
echo "🌐 Domain:"
echo "$response" | jq -r '.domain'
echo
echo "🔍 Image Query:"
echo "$response" | jq -r '.image_query // "null"'
echo
echo "🖼️  Images:"
echo "$response" | jq -r '.images[]'
echo

# ---------- Save HTML ----------
HTML_FILE="marketing_agent_output.html"
echo "$response" | jq -r '.html' > "$HTML_FILE"
echo "📄 HTML saved to $HTML_FILE"
# echo "🌐 Opening in browser..."
# xdg-open "$HTML_FILE"
