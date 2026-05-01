FROM python:3.12-alpine

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY app/ ./app/
COPY models/model_v1.pkl ./models/model_v1.pkl
COPY models/model_v2.pkl ./models/model_v2.pkl
COPY models/model_meta.json ./models/model_meta.json

# Expose port
EXPOSE 5000

# Run the service
CMD ["python", "-m", "app.api"]
