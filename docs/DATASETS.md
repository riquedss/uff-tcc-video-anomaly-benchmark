# Datasets, features e execução em GPU

Os três modelos consomem **features pré-extraídas**, não vídeo bruto. Só o
AnomalyDetection traz um extrator no repo; PEL4VAD e RTFM assumem features I3D
já prontas (os autores as disponibilizam para download).

## 1. Configuração

Copie `.env.example` para `.env` e aponte `DATA_ROOT` para o diretório do host
com os dados. Ele é montado como `/data` dentro de todos os containers —
gravável no `anomaly-detection` (que extrai features), somente leitura nos
outros dois.

```sh
cp .env.example .env
$EDITOR .env
```

## 2. Layout esperado abaixo de `DATA_ROOT`

```
$DATA_ROOT/
  ucf-crime/
    videos/                    # vídeos brutos, só para extração (AnomalyDetection)
      Abuse/ Arrest/ ... Training_Normal_Videos_Anomaly/
    features/
      mfnet/  ou  c3d/         # AnomalyDetection — .txt, 32 segmentos por vídeo
      i3d/                     # PEL4VAD — .npy, dim 1024, subdirs train/ e test/
      i3d-10crop/              # RTFM — .npy, dim 2048, subdirs train/ e test/
  shanghaitech/
    features/
      i3d/                     # PEL4VAD
      i3d-10crop/              # RTFM
```

Os nomes dos subdiretórios são convenção deste projeto — o que importa é que
batam com as variáveis dos Makefiles (`FEATURES_PATH`, `PEL4VAD_FEAT_PREFIX`,
`FEATURES_DIR`).

Dimensionamento: as features I3D do UCF-Crime passam de 100 GB. Reserve disco
antes de baixar.

## 3. Onde baixar

Os links oficiais estão nos READMEs de cada modelo — vão direto neles para não
usar URL desatualizada:

- **RTFM** — features I3D 10-crop de ShanghaiTech e UCF-Crime:
  [models/RTFM/README.md](../models/RTFM/README.md), seção de download.
- **PEL4VAD** — features I3D e homepages dos datasets:
  [models/PEL4VAD/README.md](../models/PEL4VAD/README.md), tabela de datasets.
- **AnomalyDetection** — features C3D precomputadas e pesos do extrator:
  [models/AnomalyDetectionCVPR2018-Pytorch/README.md](../models/AnomalyDetectionCVPR2018-Pytorch/README.md),
  seções *Precomputed Features* e *Feature Extractor Weights*.

Vídeos brutos do UCF-Crime: homepage do CRCV/UCF (linkada no README do PEL4VAD).

## 4. GPU

O `docker-compose.gpu.yml` reserva 1 GPU NVIDIA por serviço. A máquina precisa
de driver NVIDIA + `nvidia-container-toolkit` (o Docker deve listar o runtime
`nvidia` em `docker info`). Ative com `GPU=1`:

```sh
make GPU=1 check-gpu      # deve imprimir `cuda: True` nos três
make GPU=1 up-rtfm        # shell no container com GPU
```

Sem `GPU=1` tudo roda em CPU — os três já fazem
`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`, então caem para
CPU sozinhos, só que lentíssimo em treino real.

Versões de CUDA por imagem (relevante para compatibilidade com o driver do host):

| Serviço | Base | CUDA |
|---|---|---|
| anomaly-detection | pytorch 2.0.1 | 11.7 |
| pel4vad | pytorch 1.8.0 | 11.1 |
| rtfm | pytorch 1.7.1 | 11.0 |

## 5. Fluxo por modelo

Todos os comandos abaixo rodam **dentro** do container.

### AnomalyDetection (UCF-Crime)

```sh
make GPU=1 up-anomaly
cd AnomalyDetectionCVPR2018-Pytorch

# Opção A: extrair features dos vídeos
make extract-features            # usa MODEL_TYPE=mfnet, o peso já presente em pretrained/
# Opção B: usar features C3D precomputadas — baixe para
#   $DATA_ROOT/ucf-crime/features/c3d e passe MODEL_TYPE=c3d

make train                       # FEATURES_PATH default = /data/ucf-crime/features/$(MODEL_TYPE)
make roc MODEL_PATH=exps/models/epoch_80000.pt
```

As anotações (`Train_Annotation.txt`, `Test_Annotation.txt`) são do UCF-Crime e
já estão versionadas. Para ShanghaiTech seria preciso gerar anotações novas no
mesmo formato (`<categoria>/<video>.mp4 <n_frames>`).

### PEL4VAD

O `configs.py` tem o `feat_prefix` hardcoded no caminho dos autores; a variável
`PEL4VAD_FEAT_PREFIX` (definida no `.env`, repassada pelo compose) sobrescreve.

```sh
make GPU=1 up-pel4vad
cd PEL4VAD
make train DATASET=ucf          # ou DATASET=sh para ShanghaiTech
make infer DATASET=ucf          # usa cfg.ckpt_path (checkpoints já vêm no repo)
```

Ajuste `PEL4VAD_FEAT_PREFIX` no `.env` conforme o dataset escolhido — ele não é
derivado de `--dataset`.

### RTFM

As `.list` versionadas no repo guardam caminhos absolutos da máquina dos autores
(`/home/yu/yu_ssd/...`). Regenere-as apontando para o mount local:

```sh
make GPU=1 up-rtfm
cd RTFM
make make-lists DATASET=shanghai FEATURES_DIR=/data/shanghaitech/features/i3d-10crop
make train-local DATASET=shanghai
```

Para UCF-Crime: `DATASET=ucf FEATURES_DIR=/data/ucf-crime/features/i3d-10crop`.

`scripts/make_lists.py` só troca o diretório de cada linha, preservando a ordem
original — necessário porque `dataset.py` separa anômalo de normal por índice
fixo (`[:63]`/`[63:]` no shanghai, `[:810]`/`[810:]` no ucf) e a list de teste
precisa casar linha a linha com o `gt-*.npy`. Ele avisa quais features listadas
não foram encontradas; use `--strict` para falhar nesse caso.

O treino também sobe o dashboard visdom em `http://localhost:8097` (a porta é
publicada por `make up-rtfm`).

## 6. Validação sem dataset

```sh
make smoke        # os três, com features sintéticas
make smoke-rtfm   # ou individualmente
```

Valida imagem, dependências, dataloaders e loop de treino ponta a ponta. Não
diz nada sobre acurácia — só que o pipeline executa.
