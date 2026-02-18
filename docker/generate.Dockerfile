FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml .
COPY nothing_gpt/ nothing_gpt/
RUN pip install --no-cache-dir .
