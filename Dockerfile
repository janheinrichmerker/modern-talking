FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /workspace

RUN pip install pipenv
COPY Pipfile Pipfile.lock /workspace/
RUN pipenv install --system --deploy

COPY modern_talking/ /workspace/modern_talking/
COPY main.py /workspace/

ENTRYPOINT ["python", "main.py"]
CMD ["traineval", "distilbert-base-uncased-dropout-0.2-bilstm-64-dropout-0.2-subtract", "map-strict"]
