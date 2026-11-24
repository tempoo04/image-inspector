# Vision Dataset Explorer — container image.
# Core app only (no PyTorch/FAISS); add requirements-optional.txt for Module G.
FROM python:3.11-slim

# OpenCV headless still needs a couple of shared libs at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# The Streamlit entry point (app/main.py) lands in Milestone 2. Until then the
# CLI indexer is available:  docker run ... python -m app.launcher --scan /data
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0"]
