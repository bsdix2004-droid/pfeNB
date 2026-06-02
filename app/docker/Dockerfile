# ─────────────────────────────────────────────
#  Stage 1 – Python dependency builder
# ─────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# System deps needed to compile some ML packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      g++ \
      protobuf-compiler \
      libprotobuf-dev \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgomp1 \
      libglx0 \
      curl && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create pip config to optimize downloads
RUN mkdir -p ~/.pip && cat > ~/.pip/pip.conf << 'EOF'
[global]
timeout = 1000
retries = 5
EOF

# Copy requirements first — Docker layer cache:
# if requirements.txt doesn't change, this layer is reused.
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefix=/install --no-cache-dir \
      --default-timeout=1000 \
      --retries 5 \
      -r requirements.txt


# ─────────────────────────────────────────────
#  Stage 2 – Runtime image
# ─────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /home/appuser && chown appuser:appuser /home/appuser

WORKDIR /app

# Runtime system libs only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libgomp1 \
      libglx0 \
      libgl1 \
      curl && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Ensure packaging is available (used by skimage/sauvola preprocessing)
RUN pip install --no-cache-dir packaging

# Copy application source
COPY --chown=appuser:appuser . .

# Create directories the app will write to
RUN mkdir -p /app/logs /app/tmp /app/output/runs && chown -R appuser:appuser /app/logs /app/tmp /app/output

USER appuser

EXPOSE 8000

# Health-check used by Docker and compose depends_on
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Overridden per-environment in compose
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
