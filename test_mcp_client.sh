#!/bin/bash


# Default values
AGENT_TYPE=${1:-"orchestrator_agent"}
THREAD_ID=${2:-"test-thread-1"}
USER_QUERY=${3:-"what is 9 times 4?"}

# Log what you're sending
echo "Sending to agent:"
echo "  agent_type : $AGENT_TYPE"
echo "  thread_id  : $THREAD_ID"
echo "  user_query : $USER_QUERY"
echo ""

# Send request
curl -s -X POST -H "Content-Type: application/json" \
  -d "{\"agent_type\": \"$AGENT_TYPE\", \"thread_id\": \"$THREAD_ID\", \"user_query\": \"$USER_QUERY\"}" \
  http://localhost:8001/invoke | jq