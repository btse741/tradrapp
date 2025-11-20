# Stage 1: Builder - build dependencies and wheels
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS builder

WORKDIR /app

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Environment variables for cleaner Python behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies, build tools, python, R, and libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential libpq-dev python3 python3-pip python3-dev \
    r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels (with pip cache for speed)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

# Optional: build wheels for specific heavy packages like numba and llvmlite
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 wheel --no-cache-dir --no-deps --wheel-dir /wheels numba llvmlite

# Stage 2: Runtime - slim image
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

WORKDIR /app

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python3, pip3, runtime dependencies, and R packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and requirements
COPY --from=builder /wheels /wheels
COPY requirements.txt .

# Install all dependencies from pre-built wheels (with pip cache)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir /wheels/*

# Install Playwright and its dependencies
RUN pip3 install playwright && python3 -m playwright install chromium

# Copy your app code
COPY . /app

# Create necessary directories
RUN mkdir -p /app/data /app/output

# Make your script executable
RUN chmod +x /app/run_all.sh

# Set CUDA library path environment variable
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}

# Default command to run your script
CMD ["/app/run_all.sh"]
