# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose port (Cloud Run default is 8080)
ENV PORT=8000
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "gang:app", "--host", "0.0.0.0", "--port", "8000"]