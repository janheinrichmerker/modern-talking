FROM tensorflow/tensorflow:latest-gpu-py3
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9.
RUN apt-get update &&
    apt-get install software-properties-common &&
    add-apt-repository ppa:deadsnakes/ppa &&
    apt-get install python3.9 &&
    update-alternatives --install /usr/local/bin/python python /usr/bin/python3 40

# Set working directory.
WORKDIR /workspace

# Install Pipenv and Python dependencies.
RUN pip install pipenv
COPY Pipfile Pipfile.lock /workspace/
RUN pipenv install --system --deploy

# Copy source code.
COPY modern_talking/ /workspace/modern_talking/
COPY main.py /workspace/

# Define entry point for Docker image.
ENTRYPOINT ["python", "main.py"]
CMD ["traineval", "distilbert-base-uncased-dropout-0.2-bilstm-64-dropout-0.2-subtract", "map-strict"]
