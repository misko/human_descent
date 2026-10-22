FROM python:3.14.9-slim-trixie AS builder

ARG TORCH_BACKEND=cu126
ARG TORCH_REQUIREMENTS=requirements.lock
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
       build-essential libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev \
       libsdl2-ttf-dev libfreetype-dev libportmidi-dev libjpeg62-turbo-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip==26.2.2 certifi==2026.10.19 \
    && pip install torch==2.14.1 torchvision==0.29.1 \
       --index-url https://download.pytorch.org/whl/${TORCH_BACKEND}
WORKDIR /src
COPY pyproject.toml README.md requirements.lock requirements-cpu.lock ./
RUN pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/${TORCH_BACKEND} -r ${TORCH_REQUIREMENTS}
COPY hudes ./hudes
RUN pip install --no-deps . && pip check

FROM python:3.14.9-slim-trixie AS runtime
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 HIGH_SCORES_PATH=/data/high_scores.sqlite3 \
    HOME=/data MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
       libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 \
       libsdl2-ttf-2.0-0 libportmidi0 libgl1 libgomp1 fontconfig fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data && chown 1000:1000 /data
COPY --from=builder /opt/venv /opt/venv
# Package installation happens in the builder; the deployed service cannot
# install dependencies. Remove both copies of pip, including vendored code.
RUN /opt/venv/bin/python -m pip uninstall -y pip \
    && /usr/local/bin/python -m pip uninstall -y pip
WORKDIR /data
USER 1000:1000
EXPOSE 10000 10001
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:10001/health', timeout=3)"
CMD ["python", "-m", "hudes.websocket_server", "--device=cuda", "--model=cnn3", "--port=10000", "--run-in=process"]

FROM runtime AS test
COPY --chown=1000:1000 tests /tests
ENV SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
CMD ["python", "-m", "pytest", "/tests", "-q"]
