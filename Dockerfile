FROM registry.redhat.io/ubi9/python-311@sha256:28cd2e9c5333d4f8d2bd870c81b5bab6113ea5eff182d2bf5604923704105ef2

WORKDIR /app

USER root
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

COPY pyproject.toml __init__.py ./
COPY components/ components/
COPY pipelines/ pipelines/

RUN chown -R 1001:1001 /app
USER 1001

RUN uv sync --no-cache --extra test

CMD ["python"]