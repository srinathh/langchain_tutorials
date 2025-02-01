FROM python:3.13
# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install -e .

# Make port 5000 available to the world outside this container
# EXPOSE 5000

# Run app.py when the container launches
CMD ["langchain_tutorials"]

