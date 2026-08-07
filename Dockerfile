FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e . --no-deps

COPY dashboard.py .
COPY api_cache/pjm_load_model.pkl   api_cache/pjm_load_model.pkl
COPY api_cache/caiso_load_model.pkl api_cache/caiso_load_model.pkl
COPY api_cache/ercot_load_model.pkl api_cache/ercot_load_model.pkl
COPY api_cache/miso_load_model.pkl  api_cache/miso_load_model.pkl
COPY api_cache/nyiso_load_model.pkl api_cache/nyiso_load_model.pkl
COPY api_cache/isone_load_model.pkl api_cache/isone_load_model.pkl
COPY api_cache/spp_load_model.pkl   api_cache/spp_load_model.pkl

COPY api_cache/pjm_load_training.json   api_cache/pjm_load_training.json
COPY api_cache/caiso_load_training.json api_cache/caiso_load_training.json
COPY api_cache/ercot_load_training.json api_cache/ercot_load_training.json
COPY api_cache/miso_load_training.json  api_cache/miso_load_training.json
COPY api_cache/nyiso_load_training.json api_cache/nyiso_load_training.json
COPY api_cache/isone_load_training.json api_cache/isone_load_training.json
COPY api_cache/spp_load_training.json   api_cache/spp_load_training.json

RUN mkdir -p api_cache

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["gunicorn", "dashboard:server", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1"]
