
# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy your semantic search code
COPY ../semantic_search /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the pipeline script
CMD ["python", "run_pipeline.py"]
