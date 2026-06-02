# Use an official, lightweight Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Set environment variables:
# PYTHONDONTWRITEBYTECODE=1 prevents Python from writing .pyc files to disc
# PYTHONUNBUFFERED=1 ensures console logs are printed immediately to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy only requirements.txt first to leverage Docker's caching mechanism
COPY requirements.txt .

# Upgrade pip first, then install requirements with a 1000s timeout and 5 retries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 --retries 5 -r requirements.txt

# Copy the rest of the application files
# We need app/ (FastAPI), src/ (utilities), and models/ (saved pipelines)
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

# Expose port 8000 for the FastAPI server to listen on
EXPOSE 8000

# Start the FastAPI server using Uvicorn when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
