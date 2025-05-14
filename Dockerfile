# Import Python 3 docker
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Create user
RUN adduser --disabled-password --gecos '' myuser
USER myuser

# Run python file
ENTRYPOINT [ "python3" ]
