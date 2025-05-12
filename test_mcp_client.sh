#!/usr/bin/env bash
# test_mcp_client.sh

AGENT_TYPE=${1:-"orchestrator_agent"}
THREAD_ID=${2:-"multi-task-thread"}

# print log to stderr so jq only sees JSON on stdout
echo "Seeding tasks and invoking ${AGENT_TYPE}…" >&2

read -r -d '' TASKS << 'EOF'
[
  {
    "id": "1",
    "description": "What is 9 multiplied by 4?",
    "assigned_to": "math_agent",
    "status": "pending",
    "result": null
  },
  {
    "id": "2",
    "description": "What is the capital of France?",
    "assigned_to": "web_search_agent",
    "status": "pending",
    "result": null
  }
]
EOF

curl -s -X POST http://localhost:8001/run_tasks \
  -H "Content-Type: application/json" \
  -d "$TASKS" \
| jq