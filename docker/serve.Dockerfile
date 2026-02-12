FROM vllm/vllm-openai:v0.15.1

ENV DEBIAN_FRONTEND=noninteractive

COPY pyproject.toml /app/
COPY nothing_gpt/ /app/nothing_gpt/
WORKDIR /app
RUN pip install --no-cache-dir .

EXPOSE 8000
ENTRYPOINT []
CMD ["python3", "-m", "nothing_gpt.serve.server"]
