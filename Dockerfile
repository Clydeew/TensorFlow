FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-ind && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.7
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --extra ocr
RUN useradd --create-home --uid 1000 app && mkdir /workspace && chown app:app /workspace
ENV PATH="/app/.venv/bin:$PATH"
USER app
WORKDIR /workspace
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "/app/src/tensorforge/ui.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--theme.base=dark", "--browser.gatherUsageStats=false"]
