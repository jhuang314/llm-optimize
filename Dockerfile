FROM python:3.9-slim

# Copy the requirements file
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install -r /app/requirements.txt

# Copy over the rest of the files.
COPY main.py /app/main.py


# Set the working directory
WORKDIR /app

# Expose the port
EXPOSE 8000

# Run the command to start the server
CMD ["gunicorn", "-w", "1",  "-b", "0.0.0.0", "main:app"]
