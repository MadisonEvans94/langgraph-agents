#!/usr/bin/env bash
set -euo pipefail

PDF_PATH="sample_input.pdf"
ENDPOINT="${1:-http://127.0.0.1:8001/run_marketing_supervisor}"
HTML_FILE="marketing_agent_output.html"

[[ -f "$PDF_PATH" ]] || { echo "❌  PDF not found: $PDF_PATH" >&2; exit 1; }

printf "⏳ Waiting for agent-service"
for i in {1..20}; do curl -s -o /dev/null "$ENDPOINT" && break; printf "."; sleep 0.5; done
echo

# Single line: grab .html from JSON
html=$(curl -s -X POST -F "file=@${PDF_PATH};type=application/pdf" "$ENDPOINT" | jq -r '.html')

[[ -n "$html" ]] || { echo "⚠️  No HTML returned." >&2; exit 1; }

printf '%s\n' "$html" > "$HTML_FILE"
echo "📄 HTML saved locally to $HTML_FILE"
