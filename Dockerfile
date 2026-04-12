FROM python:3.11-slim

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY __init__.py .
COPY server/ ./server/

ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "server/app.py"]