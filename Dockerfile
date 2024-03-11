FROM python:3.12-slim

# Install Git.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y && \
    apt-get install -y git

# Install Pip.
RUN --mount=type=cache,target=/root/.cache/pip \
    ([ -d /venv ] || python -m venv /venv) && \
    /venv/bin/pip install --upgrade pip

# Set working directory.
WORKDIR /workspace/

# Install Python dependencies.
ADD pyproject.toml pyproject.toml
ARG PSEUDO_VERSION=1
RUN \
    --mount=type=cache,target=/root/.cache/pip \
    SETUPTOOLS_SCM_PRETEND_VERSION=${PSEUDO_VERSION} \
    /venv/bin/pip install -e .
RUN \
    --mount=source=.git,target=.git,type=bind \
    --mount=type=cache,target=/root/.cache/pip \
    /venv/bin/pip install -e .

# Copy source code.
COPY modern_talking/ /workspace/modern_talking/
COPY main.py /workspace/

# Define entry point for Docker image.
ENTRYPOINT ["/venv/bin/python", "main.py"]
CMD ["traineval", "distilbert-base-uncased-dropout-0.2-bilstm-64-dropout-0.2-subtract", "map-strict"]
