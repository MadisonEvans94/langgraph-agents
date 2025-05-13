#!/usr/bin/env bash
# test_supervisor.sh
# Usage: ./test_supervisor.sh "Your query here"

QUERY=${1:?Please supply a query}

echo "→ Testing supervisorAgent with query: \"$QUERY\"" >&2

curl -s -X POST http://localhost:8001/invoke \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
        --arg uq "$QUERY" \
        '{agent_type:"supervisor_agent", thread_id:null, user_query:$uq}')" \
| jq