
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 05_deploy.py .
COPY best_model.pkl .

RUN mkdir -p monitoring

EXPOSE 8000

CMD ["uvicorn", "05_deploy:app", "--host", "0.0.0.0", "--port", "8000"]