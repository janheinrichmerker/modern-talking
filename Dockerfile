FROM tensorflow/tensorflow:latest-gpu-py3
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9.
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y python3.9-full
RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python3.9 1

# Set working directory.
WORKDIR /workspace

# Install Pip, Pipenv and Python dependencies.
RUN pip install --upgrade pip && pip install pipenv
COPY Pipfile Pipfile.lock /workspace/
RUN pipenv --python 3.9
RUN pipenv install --deploy

# Copy source code.
COPY modern_talking/ /workspace/modern_talking/
COPY main.py /workspace/

# Define entry point for Docker image.
ENTRYPOINT ["python3.9", "main.py"]
CMD ["traineval", "distilbert-base-uncased-dropout-0.2-bilstm-64-dropout-0.2-subtract", "map-strict"]
