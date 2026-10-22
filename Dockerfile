FROM python:3.14.9-slim-trixie AS python-runtime

FROM ubuntu:26.04 AS pcre2-builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
       gcc make libc6-dev curl ca-certificates \
    && sed -i '/^mozilla\/emSign_Root_CA_-_G1.crt$/s/^/!/; /^mozilla\/emSign_ECC_Root_CA_-_G3.crt$/s/^/!/' /etc/ca-certificates.conf \
    && update-ca-certificates && rm -rf /var/lib/apt/lists/*
COPY docker/build-pcre2.sh /build-pcre2.sh
RUN sh /build-pcre2.sh

FROM pcre2-builder AS expat-builder
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
COPY docker/build-expat.sh /build-expat.sh
RUN sh /build-expat.sh

# Ubuntu 26.04 provides current native graphics/XML libraries. Copy the
# official CPython build; Ubuntu's newer glibc supports its binary ABI.
FROM ubuntu:26.04 AS python-base
ENV DEBIAN_FRONTEND=noninteractive PATH=/usr/local/bin:$PATH
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
       ca-certificates libbz2-1.0 libexpat1 libffi8 libgdbm-compat4t64 libgdbm6t64 \
       libncursesw6 libnsl2 libreadline8t64 libsqlite3-0 libssl3t64 libuuid1 \
       liblzma5 zlib1g libzstd1 libtinfo6 \
    && sed -i '/^mozilla\/emSign_Root_CA_-_G1.crt$/s/^/!/; /^mozilla\/emSign_ECC_Root_CA_-_G3.crt$/s/^/!/' /etc/ca-certificates.conf \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/bin/pebble
# This image is supervised by Docker and runs Python/Celery directly; the
# unmanaged Pebble init binary bundled by Ubuntu is unused.
COPY --from=python-runtime /usr/local /usr/local
RUN python3 -c "import ssl,sqlite3,ctypes,bz2,lzma,zlib,uuid,readline; assert not any('emSign Root CA - G1' in str(c['subject']) or 'emSign ECC Root CA - G3' in str(c['subject']) for c in ssl.create_default_context().get_ca_certs())"

FROM python-base AS builder

ARG TORCH_BACKEND=cu126
ARG TORCH_REQUIREMENTS=requirements.lock
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
       build-essential libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev \
       libsdl2-ttf-dev libfreetype-dev libportmidi-dev libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt PIP_CERT=/etc/ssl/certs/ca-certificates.crt
RUN pip install --upgrade pip==26.2.2
WORKDIR /src
COPY requirements.lock requirements-cpu.lock ./
RUN pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/${TORCH_BACKEND} -r ${TORCH_REQUIREMENTS}
COPY pyproject.toml README.md ./
COPY hudes ./hudes
RUN pip install --no-deps . && pip check
# Requests and explicit certifi users must share the distrusted system store.
RUN python -c "import certifi,pathlib,ssl; p=pathlib.Path(certifi.where()); p.unlink(); p.symlink_to('/etc/ssl/certs/ca-certificates.crt'); assert not any('emSign Root CA - G1' in str(c['subject']) or 'emSign ECC Root CA - G3' in str(c['subject']) for c in ssl.create_default_context(cafile=certifi.where()).get_ca_certs())"

FROM python-base AS runtime
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 HIGH_SCORES_PATH=/data/high_scores.sqlite3 \
    HOME=/data MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
       libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-mixer-2.0-0 \
       libsdl2-ttf-2.0-0 libportmidi2 libgl1 libgomp1 fontconfig fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data && chown 1000:1000 /data
COPY --from=pcre2-builder /out/*.deb /tmp/pcre2.deb
COPY --from=pcre2-builder /out/pcre2-build.json /usr/local/share/security/pcre2-build.json
RUN dpkg -i /tmp/pcre2.deb && ldconfig && rm /tmp/pcre2.deb \
    && apt-get update && apt-get install -y --no-install-recommends --allow-remove-essential \
       coreutils-from-gnu coreutils-from-uutils- rust-coreutils- \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY --from=expat-builder /out/libexpat1_*.deb /tmp/expat.deb
COPY --from=expat-builder /out/expat-build.json /usr/local/share/security/expat-build.json
RUN dpkg -i /tmp/expat.deb && ldconfig && rm /tmp/expat.deb \
    && python -c "import ctypes; lib=ctypes.CDLL('libexpat.so.1'); lib.XML_ExpatVersion.restype=ctypes.c_char_p; assert lib.XML_ExpatVersion()==b'expat_2.9.0'"
# Package installation happens in the builder; the deployed service cannot
# install dependencies. Remove both copies of pip, including vendored code.
RUN /opt/venv/bin/python -m pip uninstall -y pip \
    && /usr/local/bin/python -m pip uninstall -y pip
ARG BUILD_REVISION=unknown
LABEL org.opencontainers.image.revision=${BUILD_REVISION}
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt PIP_CERT=/etc/ssl/certs/ca-certificates.crt
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
