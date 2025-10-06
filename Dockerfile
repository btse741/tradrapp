# Stage 1: Builder stage - build and compile dependencies
FROM python:3.11-slim AS builder

# Set working directory inside container
WORKDIR /app

# Environment variables for cleaner Python behavior inside containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed to build some Python packages and R with dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential libpq-dev \
    r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install required R packages at build time
RUN R -e "install.packages(c('DBI', 'RPostgres', 'dplyr', 'simfinapi', 'yaml', 'lubridate', 'glue', 'data.table', 'magrittr', 'telegram.bot'), repos='https://cloud.r-project.org/')"

# Copy only requirements first to leverage Docker cache when code changes
COPY requirements.txt .

# Build wheels for all dependencies, speeding up final image installs
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt


# Stage 2: Final runtime stage - slim image only, no build dependencies
FROM python:3.11-slim

WORKDIR /app

# Environment variables for runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OS dependencies needed for Playwright Chromium browser and R runtime and dev libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget libnss3 libatk-bridge2.0-0 libcups2 libx11-xcb1 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libasound2 libpangocairo-1.0-0 libgtk-3-0 libxshmfence1 libxss1 \
    r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install required R packages at runtime in final image
RUN R -e "install.packages(c('DBI', 'RPostgres', 'dplyr', 'simfinapi', 'yaml', 'lubridate', 'glue', 'data.table', 'magrittr', 'telegram.bot'), repos='https://cloud.r-project.org/')"

# Copy built wheels from builder stage
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install all dependencies from wheels for fast, repeatable install
RUN pip install --no-cache-dir /wheels/*

# Install Playwright and Chromium browser
RUN pip install playwright \
    && python -m playwright install chromium

# Copy the application source code to the container
COPY . /app

# Create data and output directories expected by your scripts
RUN mkdir -p /app/data /app/output

# Ensure run_all.sh is executable
RUN chmod +x /app/run_all.sh

# Default command to execute your shell script that runs update scripts
CMD ["/app/run_all.sh"]
