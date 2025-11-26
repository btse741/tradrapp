# Stage 1: Builder - build dependencies and wheels
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system build dependencies including Python 3.10, distutils, and R dev libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential libpq-dev python3.10 python3.10-venv python3-pip python3.10-distutils \
    r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel, and build tools before building wheels
RUN python3.10 -m pip install --upgrade pip setuptools wheel build

# Set python3 alternative to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

# Stage 2: Runtime - slim image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev libpq-dev \
    && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /app/data /app/output

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python3.10 -m pip install --upgrade pip

RUN R -e "install.packages(c('DBI','RPostgres','dplyr'), repos='https://cloud.r-project.org')"

COPY --from=builder /wheels /wheels
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip install --no-cache-dir /wheels/*

RUN python3.10 -m pip install playwright && python3.10 -m playwright install chromium

COPY . /app

RUN chmod +x /app/run_all.sh

ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}

CMD ["/app/run_all.sh"]
