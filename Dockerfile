# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install required packages (bash, curl, vim) and clean up
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables to prevent Python from writing .pyc files and enable buffer output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure scripts are executable
RUN chmod +x agent_service/setup_service_env.sh agent_service/start_service.sh

# Expose the port that the FastAPI app will run on
EXPOSE 8001

# Default command to run your service
CMD ["bash", "-c", ". agent_service/setup_service_env.sh && gunicorn agent_service.app.main:app --workers ${LOCAL_REST_SERVER_NUM_WORKERS:-1} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${LOCAL_REST_SERVER_PORT:-8001} --timeout ${LOCAL_REST_SERVER_TIMEOUT:-1024}"]
