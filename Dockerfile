FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and data
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Use the working simple_app.py instead of the broken main.py
CMD ["uvicorn", "simple_app:app", "--host", "0.0.0.0", "--port", "8080"]