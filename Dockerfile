# # Use official Python image
# FROM python:3.10-slim

# # Set working directory
# WORKDIR /app

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy all source code
# COPY . .

# # Expose port (Cloud Run default is 8080)
# ENV PORT=8080
# EXPOSE 8080

# # Start the FastAPI server
# CMD ["uvicorn", "gang:app", "--host", "0.0.0.0", "--port", "8080"]

# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose the port Cloud Run expects.
# The ENV PORT=8080 is not strictly necessary as Cloud Run provides it,
# but it's good practice for local testing.
EXPOSE 8080

# FIX: Start the server by running the Python script itself.
# The script will read the $PORT environment variable and start uvicorn correctly.
# NOTE: This assumes your python file is named 'gang.py' as per your previous Dockerfile.
# If your file is named main.py, change "gang.py" to "main.py".
CMD ["python", "gang.py"]
