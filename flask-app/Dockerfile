# Build an image starting with the Python 3.8 image
FROM python:3.8

# Set environment variables
ENV HOST 0.0.0.0
ENV PORT 8080
ENV DEBUG True
ENV CELERY_BROKER redis://redis:6379/0
ENV CELERY_BACKEND redis://redis:6379/0
ENV POSTGRES_USER postgres
ENV POSTGRES_PASSWORD password
ENV POSTGRES_HOST postgres
ENV POSTGRES_PORT 5432
ENV POSTGRES_DB postgres

ENV LC_ALL en_US.utf-8
ENV LANG en_US.utf-8

# Copy all files to the directory and set it as the working directory
COPY . /flask-app
WORKDIR /flask-app

# Install dependencies
RUN pip install -r requirements.txt

# Add metadata to the image to describe that the container is listening on port 8080
EXPOSE 8080

# Run
ENTRYPOINT ["python"]
CMD ["app.py"]