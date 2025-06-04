#!/usr/bin/env bash
# End-to-end smoke test for the marketing-agent stack
set -euo pipefail

# ──────────────── Paths ────────────────
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

COMPOSE_FILE="$REPO_ROOT/docker-compose.yaml"
HEALTH_ENDPOINT="http://127.0.0.1:8001/health"
SUP_ENDPOINT="http://127.0.0.1:8001/run_marketing_supervisor"

DEFAULT_PDF="$REPO_ROOT/tests/data/sample_input.pdf"
PDF_PATH="${PDF_PATH:-$DEFAULT_PDF}"
PROMPT="${1:-Create a landing page for the attached document.}"

TIMEOUT_SEC="${TIMEOUT_SEC:-40}"
QUIET="${QUIET:-false}"

[[ -f "$PDF_PATH" ]] || { echo "PDF not found: $PDF_PATH" >&2; exit 1; }

# ───────── Build & Up ─────────
echo "Building & starting stack…"
docker compose -f "$COMPOSE_FILE" up --build -d ${QUIET:+--quiet-pull}

# Always tear down (even on Ctrl-C)
trap 'docker compose -f "$COMPOSE_FILE" down -v' EXIT

# ───────── Wait for Health ─────────
printf "Waiting for agent-service"
expiry=$((SECONDS + TIMEOUT_SEC))
until curl -sf "$HEALTH_ENDPOINT" >/dev/null; do
  [[ $SECONDS -lt $expiry ]] || { echo -e "\nTimeout waiting for health"; exit 1; }
  printf "."
  sleep 1
done

# ───────── POST Request ─────────
echo "POSTing PDF → supervisor"
resp=$(curl -s -X POST \
          -F "file=@${PDF_PATH};type=application/pdf" \
          -F "prompt=${PROMPT}" \
          "$SUP_ENDPOINT")

jq -e '.' <<<"$resp" >/dev/null || { echo "Non-JSON response"; exit 1; }

last_msg=$(jq -r '.last_message' <<<"$resp")
remote_html=$(jq -r '.html_path // empty' <<<"$resp")

echo -e "\nLast supervisor message:\n$last_msg\n"

# ───────── Copy HTML artefact ─────────
if [[ -n "$remote_html" ]]; then
  file_name=$(basename "$remote_html")
  host_html="$REPO_ROOT/agent_service/tmp/marketing_agent_outputs/$file_name"
  dest_html="$REPO_ROOT/tests/data/$file_name"

  if [[ -f "$host_html" ]]; then
    mkdir -p "$REPO_ROOT/tests/data"
    cp "$host_html" "$dest_html"
    echo "HTML copied to: $dest_html"
  else
    echo "Expected HTML not found at $host_html"
  fi
fi

echo "End-to-end test succeeded"
