# Use the Miniconda3 base image
FROM continuumio/miniconda3

# Set environment variables
ENV ENV_NAME=basic
ENV FILE_PATH=env
ENV PATH=/opt/conda/bin:$PATH

# Use bash shell for subsequent commands
SHELL ["/bin/bash", "-c"]

# Initialize Conda for bash shell and create the environment
RUN conda init bash && . ~/.bashrc && \
    conda create --name $ENV_NAME python

# Set the working directory for the web application
WORKDIR /$FILE_PATH/webapp

# Copy environment file and install dependencies
COPY . /$FILE_PATH
RUN conda run -n $ENV_NAME conda env update -n $ENV_NAME --file ../production.yml

# Install PyTorch, torchvision, and additional dependencies
RUN apt-get update && apt-get install -y libgl1 gcc && \
    conda clean --all

# Initialize Conda in the environment at runtime and download necessary files
RUN . /opt/conda/etc/profile.d/conda.sh
# Install the application; using -e will link this version onto our pythonpath instead
# of copying all our model weights and dustmaps files. This should save a few Gb.
RUN conda run -n $ENV_NAME pip install -e /$FILE_PATH
#for deployment of the server, use a manager like gunicorn rather than flask
RUN conda run -n $ENV_NAME pip install gunicorn

# Expose port 5000 for the Flask application
EXPOSE 5000

# Set the entrypoint to activate the Conda environment and run the Flask application
# the -w below is workers, so be careful to consider the resources we have alotted the server
CMD ["bash", "-c", "source activate $ENV_NAME && gunicorn -w 1 -b 0.0.0.0:5000 app:app"]
#CMD ["bash", "-c", "source activate $ENV_NAME && python -m flask run --port=5000 --host=0.0.0.0"]
