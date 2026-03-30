FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model on container startup if not present
CMD ["sh", "-c", "python model/train_model.py && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"]