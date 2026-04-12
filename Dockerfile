FROM python:3.11-slim

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY server/ ./server/
COPY __init__.py .

EXPOSE 7860

CMD ["python", "server/app.py"]