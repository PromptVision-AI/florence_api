FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy individual files
# Format: COPY <local_file> <container_destination>
COPY requirements.txt /app/
COPY main.py /app/
COPY .env /app/

# Copy directories
# Format: COPY <local_directory> <container_destination>
COPY app/ /app/app/
COPY model/ /app/model/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt


# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python3", "main.py"] 