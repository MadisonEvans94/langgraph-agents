SETUP_ENV_SCRIPT="./setup_service_env.sh"
. ${SETUP_ENV_SCRIPT}

gunicorn vllm.app.main:app --workers ${LOCAL_REST_SERVER_NUM_WORKERS} \
                                  --worker-class uvicorn.workers.UvicornWorker \
                                  --bind 0.0.0.0:${LOCAL_REST_SERVER_PORT} \
                                  --timeout ${LOCAL_REST_SERVER_TIMEOUT}