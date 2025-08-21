# Dockerfile for GPTCache with Cost-Aware Policy
FROM python:3.8-slim

WORKDIR /app

# Copy all project files
COPY . /app


# Install dependencies, force onnxruntime 1.19.0 and requests
RUN pip install --upgrade pip \
    && pip install onnxruntime==1.19.0 \
    && pip install -r requirements.txt \
    && pip install cachetools \
    && pip install requests

EXPOSE 8000

CMD ["python", "-m", "gptcache_server.server", "-s", "0.0.0.0", "-p", "8000"]
