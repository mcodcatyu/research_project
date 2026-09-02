#Official Lightweight Python base image
FROM python:3.10-slim

#Set the working directory inside the container
WORKDIR /app

# Install required system dependencies for machine learning and clear cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project source code into the container
COPY . .

#Streamlit default port
EXPOSE 8501

#Automatically launch the Stremlit app when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501","--server.address=0.0.0.0"]