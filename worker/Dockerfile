# Build an image starting with the Python 3.8 image
FROM python:3.8

# Set environment variables
ENV CELERY_BROKER redis://redis:6379/0
ENV CELERY_BACKEND redis://redis:6379/0

ENV LC_ALL en_US.utf-8
ENV LANG en_US.utf-8

# Copy all files to the directory and set it as the working directory
COPY . /worker
WORKDIR /worker

# Install dependencies
RUN pip install -r requirements.txt

# Run
ENTRYPOINT celery -A tasks worker --loglevel=info