FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY python_dash_generic.py ./
COPY IA_V7.csv ./
COPY TA_IA.csv ./
COPY V8 ./V8
COPY assets ./assets

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "--timeout", "120", "python_dash_generic:server"]
