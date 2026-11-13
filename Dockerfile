FROM python:3.11.14-slim
WORKDIR /app
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt \
    && useradd --uid 10001 --create-home app
COPY api ./api
COPY models ./models
USER 10001
ENV MODEL_PATH=/app/models/model.joblib
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
