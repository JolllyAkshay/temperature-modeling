FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
RUN pip install --no-cache-dir -e src/. --no-deps

COPY dashboard.py .
COPY api_cache/pjm_load_model.pkl api_cache/pjm_load_model.pkl
COPY api_cache/caiso_load_model.pkl api_cache/caiso_load_model.pkl

RUN mkdir -p api_cache

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["gunicorn", "dashboard:server", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
