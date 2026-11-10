FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

WORKDIR /app

# Espelho brasileiro: archive/security.ubuntu.com recusavam conexão (nov/2026).
RUN sed -i 's#http://archive.ubuntu.com#http://br.archive.ubuntu.com#g; s#http://security.ubuntu.com#http://br.archive.ubuntu.com#g' /etc/apt/sources.list \
    && apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=20 update \
    && apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=20 install -y --no-install-recommends \
        wget \
        make \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "numpy>=1.20,<1.24" \
    "visdom" \
    "scikit-learn" \
    "matplotlib" \
    "tqdm" \
    "codecarbon==2.3.4"

COPY . /app/RTFM

WORKDIR /app/RTFM

# Port used by visdom dashboard
EXPOSE 8097

# Start visdom server in background then run main
CMD ["sh", "-c", "python -m visdom.server & sleep 3 && python main.py"]
