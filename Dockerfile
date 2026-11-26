FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PARTAI_DB_PATH=/data/partai.db
WORKDIR /srv
COPY requirements.txt requirements-local.txt ./
ARG WITH_LOCAL_OCR=0
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$WITH_LOCAL_OCR" = "1" ]; then pip install --no-cache-dir -r requirements-local.txt; fi
COPY app ./app
RUN useradd -m app && mkdir /data && chown app /data
USER app
VOLUME /data
EXPOSE 8000
# Secrets come from the platform environment (OPENAI_API_KEY, SERPAPI_API_KEY), never baked into the image.
CMD ["uvicorn", "app.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
