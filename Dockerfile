# Use the lightweight Micromamba base image
FROM mambaorg/micromamba:1.5.6

# Set environment variables
ENV ENV_NAME=basic
ENV FILE_PATH=env
# Install Micromamba in a writable location
ENV MAMBA_ROOT_PREFIX=/home/mambauser/micromamba  
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

# Set the working directory to the user's home (writable)
WORKDIR /home/mambauser

# Switch to the default non-root user provided by Micromamba
USER mambauser

# Copy only the environment file first to optimize caching
COPY --chown=mambauser:mambauser production.yml /home/mambauser/production.yml

# Create the environment in the writable user directory
RUN micromamba create -y -n $ENV_NAME -c conda-forge python=3.12 && \
    micromamba install -y -n $ENV_NAME -c conda-forge --file /home/mambauser/production.yml && \
    micromamba clean --all

# Set the working directory for the web application
WORKDIR /home/mambauser/$FILE_PATH/webapp

# Copy the rest of the application (with correct permissions)
COPY --chown=mambauser:mambauser . /home/mambauser/$FILE_PATH

# Install the application using the newly created environment
RUN micromamba run -n $ENV_NAME pip install -e /home/mambauser/$FILE_PATH

# Install gunicorn for production deployment
RUN micromamba run -n $ENV_NAME pip install gunicorn

# Expose port 5000 for the Flask application
EXPOSE 5000

# Set the entrypoint to activate the environment and start the web server
ENTRYPOINT ["/bin/bash", "-c", "micromamba run -n $ENV_NAME gunicorn -w 1 -b 0.0.0.0:5000 app:app"]
