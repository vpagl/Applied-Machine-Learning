# Use an official Python runtime as a base image
FROM python:3.11-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install Flask and other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app files to the working directory
COPY app.py .
COPY score.py .
COPY Best_LR.pkl .
COPY data/ data/

# Expose the Flask app port
EXPOSE 5000

# Define the command to run the Flask app
CMD ["python3", "app.py"]
