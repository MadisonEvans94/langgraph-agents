#!/usr/bin/env bash
# End-to-end smoke test for the marketing-agent stack
set -euo pipefail

# ─────────── Path setup ───────────
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="$SCRIPT_DIR/data"
DEFAULT_PDF="$DATA_DIR/sample_input.pdf"

COMPOSE_FILE="${COMPOSE_FILE:-$REPO_ROOT/docker-compose.yaml}"
HEALTH_ENDPOINT="http://127.0.0.1:8001/health"
SUP_ENDPOINT="http://127.0.0.1:8001/run_marketing_supervisor"

PDF_PATH="${PDF_PATH:-$DEFAULT_PDF}"
PROMPT="${1:-Create a landing page for the attached document.}"

TIMEOUT_SEC="${TIMEOUT_SEC:-40}"
QUIET="${QUIET:-false}"

[[ -f "$PDF_PATH" ]] || { echo "PDF not found: $PDF_PATH" >&2; exit 1; }

# ─────────── Build & start ───────────
echo "Building and starting Docker Compose stack"
docker compose -f "$COMPOSE_FILE" up --build -d ${QUIET:+--quiet-pull}

trap 'docker compose -f "$COMPOSE_FILE" down -v' EXIT

# ─────────── Health check ───────────
printf "Waiting for agent-service"
expiry=$((SECONDS + TIMEOUT_SEC))
until curl -sf "$HEALTH_ENDPOINT" >/dev/null; do
  [[ $SECONDS -lt $expiry ]] || { echo -e "\nTimeout waiting for health"; exit 1; }
  printf "."
  sleep 1
done
echo " ready"

# ─────────── POST request ───────────
echo "Uploading PDF to marketing supervisor endpoint"
resp=$(curl -s -X POST \
          -F "file=@${PDF_PATH};type=application/pdf" \
          -F "prompt=${PROMPT}" \
          "$SUP_ENDPOINT")

jq -e '.' <<<"$resp" >/dev/null || { echo "Response is not valid JSON"; exit 1; }

last_msg=$(jq -r '.last_message' <<<"$resp")
remote_html=$(jq -r '.html_path // empty' <<<"$resp")

echo -e "\nLast supervisor message:\n$last_msg\n"

# ─────────── Copy HTML artifact ───────────
if [[ -n "$remote_html" ]]; then
  file_name=$(basename "$remote_html")
  host_html="$REPO_ROOT/agent_service/tmp/marketing_agent_outputs/$file_name"
  dest_html="$DATA_DIR/$file_name"

  if [[ -f "$host_html" ]]; then
    mkdir -p "$DATA_DIR"
    cp "$host_html" "$dest_html"
    echo "Generated HTML copied to: $dest_html"
  else
    echo "Expected HTML not found at $host_html"
  fi
fi

echo "End-to-end test completed successfully"
