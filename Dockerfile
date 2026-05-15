FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps once; cached as long as requirements.txt doesn't change
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="regression-detector" \
      org.opencontainers.image.description="LLM Model Regression Detection System" \
      org.opencontainers.image.source="https://github.com/your-org/regression-detector"


RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app


COPY --from=builder /install /usr/local


COPY src/          ./src/
COPY prompts/      ./prompts/
COPY golden_dataset/ ./golden_dataset/
COPY dashboard.py  ./dashboard.py


RUN mkdir -p /app/data && chown -R appuser:appuser /app
VOLUME ["/app/data"]


ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATABASE_URL=/app/data/evals.db \
    OPENAI_API_KEY=""

USER appuser


EXPOSE 8501


CMD ["python", "-m", "src.ci_run"]

