FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    curl \
    libpq-dev \
    bzip2 \
    ca-certificates \
    bash \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=/opt/micromamba/bin:/opt/micromamba/envs/vs/bin:${PATH}

RUN mkdir -p ${MAMBA_ROOT_PREFIX}/bin \
    && curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C ${MAMBA_ROOT_PREFIX}/bin --strip-components=1 bin/micromamba \
    && ${MAMBA_ROOT_PREFIX}/bin/micromamba create -y -n vs -c conda-forge python=3.12 pip

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install Python deps inside the 'vs' env
RUN ${MAMBA_ROOT_PREFIX}/bin/micromamba run -n vs pip install --root-user-action=ignore --no-cache-dir faiss-gpu-cu12 \
    && ${MAMBA_ROOT_PREFIX}/bin/micromamba run -n vs pip install --root-user-action=ignore --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN ${MAMBA_ROOT_PREFIX}/bin/micromamba run -n vs pip install --root-user-action=ignore --no-cache-dir -e /app

ENTRYPOINT ["/opt/micromamba/bin/micromamba", "run", "-n", "vs"]
CMD ["lit2vec", "--help"]
