# Use an official Python base image
FROM python:3.9-slim

# Set a working directory
WORKDIR /app

# Copy only necessary files for dependency installation
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY src/ /app/src
COPY .env /app/.env

# Expose the application port
EXPOSE 5000

# Run the Flask app
CMD ["python", "src/app.py"]
