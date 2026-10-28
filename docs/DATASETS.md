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

### 3.1 Status verificado dos links (2026-10-27)

Os links dos READMEs ficam desatualizados com o tempo. Testados nesta data:

| Modelo | Recurso | Status |
|---|---|---|
| RTFM | ShanghaiTech (Drive + OneDrive train/test) | ❌ 404 — todos mortos |
| RTFM | UCF-Crime (Drive train/test) | ❌ 404 — todos mortos |
| RTFM | checkpoints ShanghaiTech/UCF-Crime | ❌ 404 — todos mortos |
| PEL4VAD | ShanghaiTech I3D (`SHTech-i3d.zip`) | ✅ vivo |
| PEL4VAD | UCF-Crime I3D (OneDrive) | ✅ vivo — só precisa de download manual, ver [3.3](#33-pel4vad--ucf-crime-i3d-download-manual-via-onedrive) |
| AnomalyDetection | C3D features precomputadas (pasta Drive) | ✅ vivo |
| AnomalyDetection | pesos C3D / R3D-101 / R3D-152 | ✅ vivos |

**Conclusão prática: hoje não dá pra treinar o RTFM com os dados oficiais.**
A conta OneDrive dos autores (`uao365-my.sharepoint.com/.../a1697106_...`) não
existe mais — nenhum dos 7 links (2 datasets × train/test + 2 checkpoints +
1 Drive) resolve. Sem mirror conhecido; a alternativa seria extrair I3D
10-crop por conta própria (nenhum modelo vendorizado faz essa extração).

Primeiro momento: queremos **2 datasets por modelo** (ShanghaiTech + UCF-Crime)
→ 6 combinações. Estado real de cada uma:

| Modelo | ShanghaiTech | UCF-Crime |
|---|---|---|
| **RTFM** | ❌ sem fonte oficial | ❌ sem fonte oficial |
| **PEL4VAD** | ✅ download direto (`scripts/download-features.sh`) | ✅ download manual (ver [3.3](#33-pel4vad--ucf-crime-i3d-download-manual-via-onedrive)) |
| **AnomalyDetection** | ❌ sem features precomputadas dos autores e sem anotações no formato deles — precisaria gerar `Train/Test_Annotation.txt` novas | ✅ download direto (`scripts/download-features.sh`) |

Ou seja, hoje: 3 das 6 combinações são "baixa e roda" (as duas do PEL4VAD + o
UCF-Crime do AnomalyDetection). RTFM (2 combos) não tem fonte oficial viável;
AnomalyDetection+ShanghaiTech precisa de trabalho de anotação, não é só
download.

Use `scripts/download-features.sh` para baixar o que está vivo e é automatizável
(Google Drive / HTTP direto) direto na máquina de treino:

```sh
DATA_ROOT=/caminho/para/dados ./scripts/download-features.sh all
# ou um alvo por vez: pel4vad-sh | anomaly-c3d | anomaly-weights
```

### 3.2 Máquina Windows (faculdade)

Docker Desktop só passa GPU NVIDIA pro container via backend **WSL2** — então
todo o fluxo deste projeto (`make`, `docker compose`, o script acima) deve
rodar dentro de uma distro WSL2 (ex. Ubuntu), não no PowerShell/cmd nativo.

- Use um `DATA_ROOT` dentro do filesystem Linux do WSL2 (ex. `/home/<user>/data`),
  **não** em `/mnt/c/...` — bind mount do Docker em `/mnt/c` passa por 9p e
  fica muito mais lento com datasets grandes, além de poder falhar em
  silêncio.
- O driver NVIDIA fica instalado no Windows (host); dentro do WSL2 só precisa
  do `nvidia-container-toolkit`. `docker info` deve listar o runtime `nvidia`.
- O disco do WSL2 (`.vhdx`) cresce dinamicamente mas tem limite — confira
  espaço livre com `df -h` dentro do WSL2 antes de baixar qualquer coisa.

### 3.3 PEL4VAD + UCF-Crime I3D: download manual via OneDrive

O link do README (`models/PEL4VAD/README.md`, tabela de datasets) está vivo,
mas o SharePoint bloqueia clientes não-browser (`curl`, `gdown`, `WebFetch`)
com HTTP 403 — não é um link morto, é anti-bot. Confirmado abrindo num browser
real: a pasta carrega sem pedir login.

Conteúdo em `UCF-Crime/I3D/` (checado em 2026-10-28):

```
Train/
  RGB/                                   (pasta, 14 itens)
  RGB0.zip                               6,75 GB
  Training_Normal_Videos_Anomaly1.zip    8,85 GB
  Training_Normal_Videos_Anomaly2.zip    11,3 GB
Test/
  RGB/                                   (pasta, 14 itens)
```

~27 GB só nos zips de treino — reserve espaço (ver [2](#2-layout-esperado-abaixo-de-data_root)).

Passo a passo (não dá pra automatizar — exige JS/cookies de um browser real):

1. Abra a pasta no navegador (Windows, fora do WSL2):
   [UCF-Crime/I3D no OneDrive](https://stuxidianeducn-my.sharepoint.com/personal/pengwu_stu_xidian_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpengwu%5Fstu%5Fxidian%5Fedu%5Fcn%2FDocuments%2FUCF%2DCrime%2FI3D&ga=1)
2. Em `Train/`, baixe os 3 `.zip` (ou use "Baixar" na pasta `RGB/` se preferir
   os arquivos soltos em vez dos zips agregados — confirme que não são
   duplicados antes de baixar os dois).
3. Em `Test/`, selecione `RGB/` e clique **Baixar** — o OneDrive empacota a
   pasta num `.zip` na hora.
4. Os arquivos caem em algo como `C:\Users\<você>\Downloads\`. Mova pro
   filesystem do WSL2 (não deixe rodando direto de `/mnt/c`):
   ```sh
   mkdir -p "$DATA_ROOT/ucf-crime/features/i3d"
   cp /mnt/c/Users/<você>/Downloads/*.zip "$DATA_ROOT/ucf-crime/features/i3d/"
   cd "$DATA_ROOT/ucf-crime/features/i3d"
   for f in *.zip; do unzip -o "$f"; done
   ```
5. Confirme a estrutura final contra o layout esperado em
   [2](#2-layout-esperado-abaixo-de-data_root) (`train/` e `test/` com os
   `.npy`) antes de setar `PEL4VAD_FEAT_PREFIX=/data/ucf-crime/features/i3d`
   no `.env`.

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

## 7. Escopo do primeiro momento e limitações conhecidas

Decisão (2026-10-28): o primeiro momento do projeto roda **PEL4VAD nos dois
datasets** (ShanghaiTech + UCF-Crime) como comparação principal, mais
**AnomalyDetection em UCF-Crime**. RTFM fica de fora, e AnomalyDetection em
ShanghaiTech também — ambos por falta de dados prontos e reprodutíveis, não
por limitação do código. Investigado em profundidade antes de decidir; resumo
de cada caso abaixo.

### 7.1 RTFM — sem dataset viável

Os 7 links oficiais do README (ShanghaiTech e UCF-Crime train/test, 2
checkpoints, 1 Drive) estão todos mortos — a conta OneDrive dos autores
(`uao365-my.sharepoint.com/.../a1697106_...`) não existe mais (ver [3.1](#31-status-verificado-dos-links-2026-10-27)).
Não há mirror conhecido (checado em GitHub issues do repo oficial e forks,
Kaggle, Hugging Face, Papers With Code).

A alternativa seria extrair I3D 10-crop por conta própria:

- O extrator original dos autores (`Tushar-N/pytorch-resnet3d`) está
  abandonado, sem pesos prontos (precisa converter `.pkl` do Caffe2/`video-nonlocal-net`
  da Meta), e foi testado num ambiente antigo (Python 3.7 / PyTorch 1.1).
- A alternativa mantida (`i3dFeatureExtraction`, PyPI) não tem garantia de
  reproduzir exatamente o formato que `models/RTFM/dataset.py` espera (dim
  2048, 10-crop, janela/stride de segmento) — risco real de treinar com
  features incompatíveis sem forma fácil de detectar o problema.
- `data/split-shanghaitech` (dataset bruto já baixado neste repo) parece
  conter só o split de **teste** do ShanghaiTech original — o Campus Dataset
  oficial ainda está no ar, mas os links de download da própria página
  (`svip-lab.github.io/dataset/campus_dataset.html`) estão quebrados/bloqueados.

**Conclusão: extrair I3D pro RTFM é um mini-projeto à parte (semanas, não
horas), com risco de reprodutibilidade. Fica fora do primeiro momento.**
Revisitar só se sobrar tempo depois dos outros dois modelos estarem rodando,
e como extração própria, não como download.

### 7.2 AnomalyDetection + ShanghaiTech — falta engenharia de anotação, não download

O split weakly-supervised correto existe e é conhecido: Zhong et al. (CVPR
2019, *Graph Convolutional Label Noise Cleaner*), publicado em
[`GCN-Anomaly-Detection/ShanghaiTech_new_split`](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection) —
`SH_Train.txt` (238 vídeos) e `SH_Test.txt` (199 vídeos). É o mesmo split que
o RTFM usa internamente (`[:63]`/`[63:]`). Mas são só listas de IDs de vídeo,
sem rótulo normal/anômalo nem GT temporal.

Lendo `features_loader.py` do AnomalyDetection localmente:

- No **treino**, a anotação só separa Normal/Anomaly pela palavra `"Normal"`
  no caminho do arquivo — a coluna de contagem de frames não é lida.
- No **teste**, a anotação carrega **GT temporal real** (até 2 pares de
  intervalo de frames anômalos), usado no cálculo do ROC — não é cosmético.

`data/split-shanghaitech` (já baixado neste repo) inclui os vídeos de
**treino** também (`training/videos/*.avi`), não só o split de teste como se
pensava inicialmente — então não falta baixar mais dataset bruto. Mas falta:

1. Cruzar os IDs do split de Zhong com a origem treino/teste original de cada
   vídeo, pra classificar Normal/Anomaly.
2. Derivar os intervalos de GT temporal a partir das máscaras pixel-level que
   já temos — o formato do loader só suporta 2 intervalos por vídeo, pode
   exigir mesclar segmentos ou editar o loader pra vídeos com mais anomalias.
3. Extrair features C3D/MFNet pros ~437 vídeos do split (pipeline já existe —
   `make extract-features` — mas é custo computacional real).

**Conclusão: é mais fácil que o RTFM (não precisa reinventar extrator), mas
ainda é trabalho de engenharia de dados, não um passo de setup.** Fica fora
do primeiro momento; revisitar depois se o projeto precisar da comparação
completa nos 3 modelos × 2 datasets.
