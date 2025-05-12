#!/usr/bin/env bash
# test_planner.sh
# Usage: ./test_planner.sh "Your query here"

QUERY=${1:?Please provide a query, e.g. "Find 9*4 and the capital of France"}

# Log to stderr so jq only sees the JSON response
echo "→ Testing PlanningAgent with query: \"$QUERY\"" >&2

curl -s -X POST http://localhost:8001/invoke \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
        --arg uq "$QUERY" \
        '{agent_type:"planning_agent", thread_id:null, user_query:$uq}')" \
| jq