# Use an official Python 3.10.4 image as the base
FROM python:3.10.4-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container's working directory
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Define environment variables, if needed (optional)
# ENV ENV_VAR_NAME=value

# Run the application
CMD ["python", "run_3.py"]
