# ─────────────────────────────────────────────────────────────
# Dockerfile - Student Dropout Predictor
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create plots directory
RUN mkdir -p plots

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entrypoint: train model if artifacts don't exist, then launch app
CMD ["sh", "-c", "\
    if [ ! -f xgboost_model.pkl ]; then \
        echo '=== Training model for the first time ===' && \
        python train_model.py; \
    else \
        echo '=== Model artifacts found, skipping training ==='; \
    fi && \
    streamlit run app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false \
    "]
