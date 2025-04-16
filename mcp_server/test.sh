#!/bin/bash
# query.sh - Rapid prototyping queries to the agent REST API

echo "Sending Query 1"
curl -N -X POST -H "Content-Type: application/json" \
  -d '{"agent_type": "mcp_agent", "thread_id": "1a543", "user_query": "what did I just ask?"}' \
  http://localhost:8001/ask
echo -e "\nQuery 1 completed.\n"
