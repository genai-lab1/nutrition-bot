# Use a slim, stable Python image matching your working cloud environment
FROM python:3.12-slim

# Install system dependencies required for handling binaries safely
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirement files first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install dependencies using standard pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and the cleaned data files
COPY . .

# Expose the default port Cloud Run expects (8080)
EXPOSE 8080

# Configure Streamlit to run perfectly on Cloud Run infrastructure
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]