# Use the official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire current directory contents into the container at /app
COPY . /app

# Install necessary dependencies
RUN pip install --no-cache-dir torch flask flask_cors langchain bs4 PyPDF2 requests mechanicalsoup sentence_transformers ctransformers timeout_decorator faiss-cpu

ENV PORT=5000

EXPOSE 5000

CMD ["python", "model.py"]