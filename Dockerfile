FROM python:3.12-slim

# opencv-python needs libGL/libglib at import time even in a headless
# container that never opens a window.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .

COPY synthetic_generator.py ./
COPY configs/ ./configs/
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
