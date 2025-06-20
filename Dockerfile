# Generated by https://smithery.ai. See: https://smithery.ai/docs/build/project-config
FROM python:3.12-slim

# Set unbuffered output
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command
CMD ["python", "mlflow_server.py"]
