# Use a Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app


# Upgrade pip to avoid resolver quirks with older pip versions
RUN python -m pip install --upgrade pip


# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY app.py .
#COPY config.json .
COPY customer_orders.db .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
