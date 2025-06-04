# agent_service/Dockerfile

FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
        bash curl vim \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_service ./agent_service
COPY agent_resources ./agent_resources
COPY config.yaml ./

RUN chmod +x agent_service/setup_service_env.sh \
            agent_service/start_service.sh

EXPOSE 8001

CMD ["bash", "-c", ". agent_service/setup_service_env.sh && \
     gunicorn agent_service.app.main:app \
       --workers ${LOCAL_REST_SERVER_NUM_WORKERS:-1} \
       --worker-class uvicorn.workers.UvicornWorker \
       --bind 0.0.0.0:${LOCAL_REST_SERVER_PORT:-8001} \
       --timeout ${LOCAL_REST_SERVER_TIMEOUT:-1024}"]
