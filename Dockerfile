FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POSEARCH_CONFIG=/config/config.yaml \
    POSEARCH_MODEL_LOCK=/app/models.lock.json \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.lock.txt \
    && pip install --no-cache-dir --no-deps "git+https://github.com/Tau-J/rtmlib.git@03a1693e59e4f7cd84582c0fb30459b3bf18ad42" \
    && pip install --no-cache-dir --no-deps .

COPY models.lock.json ./models.lock.json

RUN useradd --create-home --uid 10001 posesearch \
    && mkdir -p /data /models /media/library /config \
    && chown -R posesearch:posesearch /data
USER posesearch

EXPOSE 8080
CMD ["python", "-m", "posesearch.supervisor"]
