#!/usr/bin/env bash
# test_composite.sh
# Usage: ./test_composite.sh "Your query here"

QUERY=${1:?Please supply a query}

echo "→ Testing CompositeAgent with query: \"$QUERY\"" >&2

curl -s -X POST http://localhost:8001/invoke \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
        --arg uq "$QUERY" \
        '{agent_type:"composite_agent", thread_id:null, user_query:$uq}')" \
| jq