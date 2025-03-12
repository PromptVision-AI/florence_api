FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the model and application code
COPY . .

# Set environment variables
ENV MODEL_PATH=/app/Florence-2-large

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python3", "app.py"] 