# Use the lightweight Micromamba base image
FROM mambaorg/micromamba:2.0-ubuntu22.04

# Switch to the default non-root user provided by Micromamba

# Set environment variables
ENV ENV_NAME=basic
USER root

# Update packages and Install packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y git python3 python3-pip 

# Set the working directory to the user's home (writable)
WORKDIR /home/mambauser

# Copy only the environment file first to optimize caching
COPY --chown=mambauser:mambauser production.yml /home/mambauser/production.yml

# Switch to the default non-root user provided by Micromamba
USER mambauser

# Create the environment in the writable user directory
# Create the environment and install packages in the writable user directory
RUN micromamba create -y -n $ENV_NAME && \
    micromamba install -y -n $ENV_NAME -c conda-forge --file /home/mambauser/production.yml && \
    micromamba clean --all && \
    micromamba clean --force-pkgs-dirs -y

# Copy the rest of the application (with correct permissions)
COPY --chown=mambauser:mambauser . /home/mambauser/env

# Install the application using the newly created environment
RUN micromamba run -n $ENV_NAME pip install -e /home/mambauser/env

#finally install gunicorn under the same environment
RUN micromamba run -n $ENV_NAME pip install gunicorn

# Set the working directory to the user's home (writable)
WORKDIR /home/mambauser/env/webapp

# Expose port 5000 for the Flask application
EXPOSE 5000

# Set the entrypoint to activate the environment and start the web server
ENTRYPOINT ["/bin/bash", "-c", "micromamba run -n $ENV_NAME gunicorn -w 1 -b 0.0.0.0:5000 app:app"]
