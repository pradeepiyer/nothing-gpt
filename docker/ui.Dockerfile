FROM python:3.13-slim

RUN pip install --no-cache-dir \
    "gradio~=5.7.1" \
    "requests" \
    "openai>=1.0"

COPY pyproject.toml /app/
COPY nothing_gpt/ /app/nothing_gpt/
WORKDIR /app
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["python", "-m", "nothing_gpt.ui.app"]
