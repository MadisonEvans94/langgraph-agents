#!/usr/bin/env bash
set -euo pipefail

PDF_PATH="sample_input.pdf"
ENDPOINT="${1:-http://127.0.0.1:8001/run_marketing_supervisor}"
PROMPT="${2:-Create a landing page for the attached document.}"
HTML_FILE="marketing_agent_output.html"

[[ -f "$PDF_PATH" ]] || { echo "❌  PDF not found: $PDF_PATH" >&2; exit 1; }

printf "⏳ Waiting for agent-service"
for i in {1..20}; do curl -s -o /dev/null "$ENDPOINT" && break; printf "."; sleep 0.5; done
echo

json=$(curl -s -X POST \
        -F "file=@${PDF_PATH};type=application/pdf" \
        -F "prompt=${PROMPT}" \
        "$ENDPOINT")

# always print the last assistant message
last_msg=$(echo "$json" | jq -r '.last_message')
echo -e "\n🗣  Last agent message:\n$last_msg\n"

# optionally save HTML if present
html=$(echo "$json" | jq -r '.html // empty')
if [[ -n "$html" ]]; then
  printf '%s\n' "$html" > "$HTML_FILE"
  echo "📄 HTML saved to $HTML_FILE"
fi
