#!/bin/bash

# Add troubleshooting logs
echo "Starting service: $(date)"
echo "Current directory: $(pwd)"
echo "Checking if setup script exists..."
if [[ ! -f "./agent_service/setup_service_env.sh" ]]; then
  echo "ERROR: setup_service_env.sh not found!"
  exit 1
fi

# Run setup script
SETUP_ENV_SCRIPT="./agent_service/setup_service_env.sh"
echo "Running setup script: $SETUP_ENV_SCRIPT"
. ${SETUP_ENV_SCRIPT}

# Check environment variables
echo "Environment variables:"
env

# Start gunicorn
echo "Starting gunicorn..."
exec gunicorn agent_service.app.main:app --workers ${LOCAL_REST_SERVER_NUM_WORKERS:-1} \
       --worker-class uvicorn.workers.UvicornWorker \
       --bind 0.0.0.0:${LOCAL_REST_SERVER_PORT:-8001} \
       --timeout ${LOCAL_REST_SERVER_TIMEOUT:-1024}
