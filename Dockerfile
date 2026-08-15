FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir \
    torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application
COPY app ./app

# Copy data
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]