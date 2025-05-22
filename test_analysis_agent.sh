#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
PDF_PATH="agent_resources/agents/marketing_agent/sample_input.pdf"
ENDPOINT="${1:-http://127.0.0.1:8001/summarize_pdf}"

# ---------- Sanity checks ----------
if [[ ! -f "$PDF_PATH" ]]; then
  echo "❌  PDF not found: $PDF_PATH" >&2
  exit 1
fi

# Optional: wait for the service to come up
printf "⏳ Waiting for agent-service"
for i in {1..20}; do
  if curl -s -o /dev/null "$ENDPOINT"; then break; fi
  printf "."
  sleep 0.5
done
echo

# ---------- Upload + print summary ----------
echo "📤  Uploading $PDF_PATH → $ENDPOINT"
curl -s -X POST \
     -F "file=@${PDF_PATH};type=application/pdf" \
     "$ENDPOINT" \
  | jq -r '.summary'

echo
echo "✅  Done."