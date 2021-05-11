FROM python:3.8

WORKDIR /workspace

RUN pip install pipenv
COPY Pipfile Pipfile.lock /workspace/
RUN pipenv install --system --deploy

# TODO Build project source and dependencies.

VOLUME /workspace/results.ipynb
CMD jupyter notebook --ip=0.0.0.0 --no-browser --allow-root /workspace/results.ipynb