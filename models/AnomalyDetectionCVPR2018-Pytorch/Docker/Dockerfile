FROM continuumio/miniconda3

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

RUN conda create -n anomaly python=3.10 -y

ENV PATH=/opt/conda/envs/anomaly/bin:$PATH

# Copy local source code (build context = AnomalyDetectionCVPR2018-Pytorch/)
COPY . /app/AnomalyDetectionCVPR2018-Pytorch

WORKDIR /app/AnomalyDetectionCVPR2018-Pytorch

RUN pip install --no-cache-dir --only-binary av -r requirements.txt

RUN chmod +x /app/AnomalyDetectionCVPR2018-Pytorch/Docker/setup_anomaly.sh

# Create non-root user with sudo restricted to apt-get only
RUN useradd -m -s /bin/bash anomalyuser && \
    echo "anomalyuser ALL=(ALL) NOPASSWD: /usr/bin/apt-get" >> /etc/sudoers && \
    chown -R anomalyuser:anomalyuser /app /opt/conda/envs/anomaly

USER anomalyuser

RUN echo "source activate anomaly" >> ~/.bashrc

# Pretrained weights and example_videos should be mounted via volumes (see docker-compose.yml)
CMD ["/bin/bash"]
