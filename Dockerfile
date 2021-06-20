FROM tensorflow/tensorflow:latest-gpu-py3

# Install Pyenv and Python 3.9.
RUN apt-get update && \
    apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget ca-certificates curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev mecab-ipadic-utf8 git
RUN git clone --branch v2.0.1 --depth 1 https://github.com/pyenv/pyenv.git /.pyenv
WORKDIR /.pyenv
ENV PYENV_ROOT /.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN src/configure && make -C src
RUN pyenv update && \
    pyenv install 3.9 && \
    pyenv global 3.9 && \
    pyenv rehash

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
