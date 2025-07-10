# Use the official Python image from the Docker Hub
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port the application will run on
EXPOSE 80

# Add python path to include src as a module
ENV PYTHONPATH "${PYTHONPATH}:/app/"

# Specify the command to run the application using Uvicorn directly
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]