# Use the lightweight Micromamba base image
FROM mambaorg/micromamba:2.0-ubuntu22.04

# Set environment variables
ENV ENV_NAME=basic

# Update packages and Install packages
USER root

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y git python3 python3-pip wget git-lfs

WORKDIR /env/mantis_shrimp/dustmaps/csfd/

RUN wget -O csfd_ebv.fits https://zenodo.org/record/8207175/files/csfd_ebv.fits && \
    wget -O /mask.fits https://zenodo.org/record/8207175/files/mask.fits

WORKDIR /env/mantis_shrimp/dustmaps/csfd/planck/
RUN wget -O HFI_CompMap_ThermalDustModel_2048_R1.20.fits "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_CompMap_ThermalDustModel_2048_R1.20.fits"

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
    micromamba run -n $ENV_NAME pip install gunicorn && \
    micromamba update && micromamba clean --all && micromamba clean --force-pkgs-dirs -y

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
# ENTRYPOINT ["/bin/bash", "-c", "micromamba run -n basic gunicorn -w 1 -b 0.0.0.0:5000 app:app"]
