FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py

RUN mkdir -p /app/data /app/storage

EXPOSE 7860
CMD ["python", "app.py"]

