# Use Python slim
FROM python:3.8.5-slim

# Set working directory
WORKDIR /app
COPY . /app

# Install dependencies
RUN apt-get update && apt-get install ffmpeg -y
RUN pip install -r requirements.txt
