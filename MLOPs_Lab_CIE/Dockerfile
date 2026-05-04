FROM python:3.11-slim

WORKDIR /app

# Copy everything into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command (CLI predictor)
ENTRYPOINT ["python", "src/predict_cli.py"]