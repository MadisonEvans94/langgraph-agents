#!/bin/bash
# query.sh - Rapid prototyping queries to the agent REST API

echo "Sending Query 1"
curl -N -X POST -H "Content-Type: application/json" \
  -d '{"agent_type": "mcp_agent", "user_query": "what is 5 + 8?"}' \
  http://localhost:8001/ask_stream
echo -e "\nQuery 1 completed.\n"

# sleep 2

# echo "Sending Query 2: Please use your multiply tool to multiply 4 and 7."
# curl -N -X POST -H "Content-Type: application/json" \
#   -d '{"agent_type": "mcp_agent", "user_query": "Please use your multiply tool to multiply 4 and 7."}' \
#   http://localhost:8001/ask_stream
# echo -e "\nQuery 2 completed.\n"
