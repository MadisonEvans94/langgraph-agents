#!/bin/bash

# LLM QnA Service
export HOME_ROOT=${HOME_ROOT:-"/app"}
export LOCAL_SERVICE_NAME="agent_service"
export LOCAL_SERVICE_DIR="${HOME_ROOT}/agent_service"
export LOCAL_REST_SERVER_PORT=${LOCAL_REST_SERVER_PORT:-"8001"}
export LOCAL_REST_SERVER_NUM_WORKERS=${LOCAL_REST_SERVER_NUM_WORKERS:-"1"}
export LOCAL_REST_SERVER_TIMEOUT=${LOCAL_REST_SERVER_TIMEOUT:-"1024"}

# LLM Parameters
export LLM_ID=${LLM_ID:-"meta-llama/Llama-3.1-8B-Instruct"}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-"1000"}
export TEMPERATURE=${TEMPERATURE:-"0.2"}
export TOP_P=${TOP_P:-"0.95"}
export REPETITION_PENALTY=${REPETITION_PENALTY:-"1.0"}
export VLLM_DOWNSTREAM_HOST=${VLLM_DOWNSTREAM_HOST:-"http://vllm-downstream:4882"}

# Echo env variables
echo "Environment Variables:"
echo "HOME_ROOT=${HOME_ROOT}"
echo "LOCAL_SERVICE_NAME=${LOCAL_SERVICE_NAME}"
echo "LOCAL_REST_SERVER_PORT=${LOCAL_REST_SERVER_PORT}"
echo "LOCAL_REST_SERVER_NUM_WORKERS=${LOCAL_REST_SERVER_NUM_WORKERS}"
echo "LOCAL_REST_SERVER_TIMEOUT=${LOCAL_REST_SERVER_TIMEOUT}"
echo "LLM_ID=${LLM_ID}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "TEMPERATURE=${TEMPERATURE}"
echo "TOP_P=${TOP_P}"
echo "REPETITION_PENALTY=${REPETITION_PENALTY}"

# Verify the essential directories (optional)
if [[ ! -d "${LOCAL_SERVICE_DIR}" ]]; then
  echo "ERROR: Service directory not found: ${LOCAL_SERVICE_DIR}"
  exit 1
fi
