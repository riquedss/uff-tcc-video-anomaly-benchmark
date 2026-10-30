#!/usr/bin/env bash
# Baixa as features/pesos pre-prontos disponiveis para os modelos deste repo.
# Rodar na maquina onde o treino vai acontecer (ex. maquina da faculdade),
# ANTES de `make GPU=1 up-<modelo>`.
#
# Maquina Windows: Docker Desktop so passa GPU NVIDIA pro container via
# backend WSL2 -- entao TODO o fluxo deste projeto (make, docker compose,
# este script) deve rodar dentro de uma distro WSL2 (ex. Ubuntu), nao no
# PowerShell/cmd nativo. Dentro do WSL2:
#   - Use um DATA_ROOT dentro do filesystem Linux (ex. /home/<user>/data),
#     NAO em /mnt/c/... -- bind mount do Docker em /mnt/c e muito mais lento
#     (passa por 9p) e pode falhar silenciosamente com datasets grandes.
#   - O disco do WSL2 (.vhdx) cresce dinamicamente mas tem limite; confirme
#     espaco livre com `df -h` dentro do WSL2 antes de baixar.
#   - driver NVIDIA fica instalado no Windows (host); dentro do WSL2 so
#     precisa do nvidia-container-toolkit. `docker info` deve listar o
#     runtime `nvidia`.
#
# Uso:
#   DATA_ROOT=/caminho/para/dados ./scripts/download-features.sh [pel4vad-sh|anomaly-c3d|anomaly-weights|all]
#
# Layout final (ver docs/DATASETS.md):
#   $DATA_ROOT/
#     shanghaitech/features/i3d/           <- PEL4VAD (SHTech-i3d.zip)
#     ucf-crime/features/c3d/              <- AnomalyDetection (precomputed C3D)
#     anomaly-weights/c3d.pickle           <- AnomalyDetection (pesos do extrator C3D)
#     anomaly-weights/r3d101.pt, r3d152.pt <- AnomalyDetection (pesos alternativos)
#
# NAO baixa nada do RTFM: os 7 links oficiais no README do RTFM (ShanghaiTech
# e UCF-Crime, train/test/checkpoints) estao todos mortos (404) a partir de
# 2026-10-27 -- a conta OneDrive dos autores nao existe mais. Ver docs/DATASETS.md.

set -euo pipefail

: "${DATA_ROOT:?defina DATA_ROOT apontando para o diretorio de dados}"
TARGET="${1:-all}"

if [[ "$DATA_ROOT" == /mnt/c/* || "$DATA_ROOT" == /mnt/[a-z]/* ]]; then
  echo "AVISO: DATA_ROOT ($DATA_ROOT) esta em /mnt/<drive>, ou seja, no disco" >&2
  echo "Windows via WSL2 -- bind mount do Docker fica lento (9p) com datasets" >&2
  echo "grandes. Prefira um caminho dentro do filesystem Linux, ex. \$HOME/data." >&2
fi

for bin in unzip wget python3; do
  command -v "$bin" >/dev/null 2>&1 || { echo "Instalando $bin..."; sudo apt-get update -y && sudo apt-get install -y "$bin"; }
done

command -v gdown >/dev/null 2>&1 || pip3 install --user gdown
export PATH="$PATH:$HOME/.local/bin"

mkdir -p "$DATA_ROOT"

download_pel4vad_sh() {
  echo "== PEL4VAD: ShanghaiTech I3D features =="
  local dir="$DATA_ROOT/shanghaitech/features"
  mkdir -p "$dir"
  gdown 1kIv502RxQnMer-8HB7zrU_GU7CNPNNDv -O "$dir/SHTech-i3d.zip"
  unzip -o "$dir/SHTech-i3d.zip" -d "$dir/i3d"
  echo "Baixado em $dir/i3d -- CONFIRME a estrutura interna (train/test) antes"
  echo "de apontar PEL4VAD_FEAT_PREFIX no .env; o zip pode extrair com um"
  echo "subdiretorio extra (ex. $dir/i3d/SHTech-i3d/...)."
}

download_anomaly_c3d() {
  echo "== AnomalyDetection: C3D features precomputadas (UCF-Crime) =="
  local dir="$DATA_ROOT/ucf-crime/features/c3d"
  mkdir -p "$dir"
  gdown --folder 1rhOuAdUqyJU4hXIhToUnh5XVvYjQiN50 -O "$dir"
}

download_anomaly_weights() {
  echo "== AnomalyDetection: pesos dos extratores =="
  local dir="$DATA_ROOT/anomaly-weights"
  mkdir -p "$dir"
  wget -c -O "$dir/c3d.pickle" "http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle"
  gdown 1p80RJsghFIKBSLKgtRG94LE38OGY5h4y -O "$dir/r3d101.pt"
  gdown 1irIdC_v7wa-sBpTiBlsMlS7BYNdj4Gr7 -O "$dir/r3d152.pt"
}

case "$TARGET" in
  pel4vad-sh) download_pel4vad_sh ;;
  anomaly-c3d) download_anomaly_c3d ;;
  anomaly-weights) download_anomaly_weights ;;
  all)
    download_pel4vad_sh
    download_anomaly_c3d
    download_anomaly_weights
    ;;
  *)
    echo "Alvo desconhecido: $TARGET" >&2
    echo "Use: pel4vad-sh | anomaly-c3d | anomaly-weights | all" >&2
    exit 1
    ;;
esac

echo "Pronto. Revise o layout em $DATA_ROOT contra docs/DATASETS.md antes de treinar."
