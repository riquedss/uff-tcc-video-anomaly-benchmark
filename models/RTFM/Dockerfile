FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        make \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "numpy>=1.20,<1.24" \
    "visdom" \
    "scikit-learn" \
    "matplotlib" \
    "tqdm"

COPY . /app/RTFM

WORKDIR /app/RTFM

# Port used by visdom dashboard
EXPOSE 8097

# Start visdom server in background then run main
CMD ["sh", "-c", "python -m visdom.server & sleep 3 && python main.py"]
