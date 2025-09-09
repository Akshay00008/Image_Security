# Use a Python base image
FROM python:3.10-slim

# Set environment variables to avoid buffering logs
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (for OpenCV and other dependencies)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 8501 (the default port for Streamlit)
EXPOSE 8501

# Set the command to run the facenet.py script with Streamlit
CMD ["streamlit", "run", "facenet.py"]
