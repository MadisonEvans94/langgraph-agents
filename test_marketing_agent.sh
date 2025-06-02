#!/usr/bin/env bash
set -euo pipefail

PDF_PATH="sample_input.pdf"
ENDPOINT="http://127.0.0.1:8001/run_marketing_supervisor"
PROMPT="${1:-Create a landing page for the attached document.}"

[[ -f "$PDF_PATH" ]] || { echo "PDF not found: $PDF_PATH" >&2; exit 1; }

printf "Waiting for agent-service"
for i in {1..20}; do
  curl -s -o /dev/null "$ENDPOINT" && break
  printf "."
  sleep 0.5
done
echo

json=$(curl -s -X POST \
        -F "file=@${PDF_PATH};type=application/pdf" \
        -F "prompt=${PROMPT}" \
        "$ENDPOINT")

last_msg=$(echo "$json" | jq -r '.last_message')
echo -e "\nLast agent message:\n$last_msg\n"

html_path=$(echo "$json" | jq -r '.html_path // empty')
if [[ -n "$html_path" ]]; then
  echo "HTML was saved by the agent at: $html_path"
  # Open in browser if you like (uncomment for Linux/macOS):
  # xdg-open "$html_path" >/dev/null 2>&1 &
fi
