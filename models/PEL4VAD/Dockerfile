FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        wget \
        ffmpeg \
        pkg-config \
        gcc \
        g++ \
        make \
        libsm6 \
        libxext6 \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libavfilter-dev \
        libswscale-dev \
        libswresample-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# av é instalado separado com --only-binary para forçar uso da wheel pré-compilada
# evita compilação do source: o sdist da 9.2.0 só compila com Cython < 3
RUN pip install --no-cache-dir --only-binary=av "av==12.3.0"

RUN pip install --no-cache-dir \
    "einops==0.6.0" \
    "ftfy==6.1.1" \
    "fvcore==0.1.5.post20220414" \
    "gensim==4.1.2" \
    "h5py==3.7.0" \
    "imageio==2.16.1" \
    "imageio-ffmpeg==0.4.5" \
    "imutils==0.5.4" \
    "iopath==0.1.9" \
    "matplotlib==3.5.0" \
    "moviepy==1.0.3" \
    "numpy==1.21.4" \
    "opencv-python-headless==4.5.5.64" \
    "pandas==1.3.4" \
    "Pillow==8.4.0" \
    "protobuf==3.20.0" \
    "psutil==5.9.0" \
    "regex==2022.9.13" \
    "scikit-learn==1.0.1" \
    "scipy==1.7.2" \
    "seaborn==0.12.0" \
    "tabulate==0.8.9" \
    "tqdm==4.63.0" \
    "wandb==0.12.14" \
    "xlwt==1.3.0" \
    "yacs==0.1.8"

COPY . /app/PEL4VAD
RUN pip install --no-cache-dir -e /app/PEL4VAD/prompt_extract

WORKDIR /app/PEL4VAD

CMD ["python", "main.py", "--dataset", "ucf", "--mode", "train"]
