# Detecção de Anomalias em Vídeo — TCC

Comparação de três modelos de Video Anomaly Detection (VAD) fracamente
supervisionado, todos vendorizados neste repositório e conteinerizados para
rodar com um único dataset externo em comum:

- **[RTFM](models/RTFM)** — Robust Temporal Feature Magnitude Learning.
- **[PEL4VAD](models/PEL4VAD)** — Prompt-Enhanced Learning for VAD.
- **[AnomalyDetectionCVPR2018-Pytorch](models/AnomalyDetectionCVPR2018-Pytorch)** — MIL sobre C3D/MFNet (Sultani et al., CVPR 2018).

Os três consomem **features de vídeo pré-extraídas** (I3D ou C3D/MFNet), não
vídeo bruto — só o AnomalyDetection traz um extrator de features no repo.
Cada modelo roda em seu próprio container Docker, com GPU opcional via
`docker-compose.gpu.yml`, todos lendo do mesmo diretório de dados montado em
`/data`.

Detalhes de layout de dados, onde baixar cada feature set e o fluxo completo
de cada modelo estão em [docs/DATASETS.md](docs/DATASETS.md). Este README foca
em colocar o **PEL4VAD** rodando primeiro — é o caminho mais curto até um
resultado real na GPU.

> **Nota de escopo (2026-10-28):** o RTFM está vendorizado no repo mas **sem
> dataset viável no momento** — os 7 links oficiais de download dos autores
> (ShanghaiTech e UCF-Crime) estão todos mortos, sem mirror conhecido, e
> extrair as features por conta própria é um mini-projeto à parte com risco
> de reprodutibilidade. O primeiro momento do projeto roda **PEL4VAD nos dois
> datasets** + **AnomalyDetection em UCF-Crime**. Detalhes e investigação
> completa em [docs/DATASETS.md §7](docs/DATASETS.md#7-escopo-do-primeiro-momento-e-limitações-conhecidas).

## Por que começar pelo PEL4VAD

Dos três, é o único com os dois datasets de interesse (ShanghaiTech e
UCF-Crime) prontos pra baixar:

- Dataset menor pra começar: **ShanghaiTech** (features I3D) é bem menor que
  o UCF-Crime, que passa de 25 GB só nas features.
- Não precisa extrair feature nenhuma (ao contrário do AnomalyDetection).
- Um único passo de preparação — baixar e apontar `PEL4VAD_FEAT_PREFIX` — e
  depois `make train`.
- Já tem smoke test validado (`make -f Makefile.smoke smoke-pel4vad`) rodando
  o pipeline inteiro com features sintéticas.

AnomalyDetection (UCF-Crime) fica para depois, seguindo os mesmos passos
descritos em [docs/DATASETS.md](docs/DATASETS.md).

## Pré-requisitos

- Docker + Docker Compose.
- Driver NVIDIA + `nvidia-container-toolkit` na máquina com GPU (o runtime
  `nvidia` precisa aparecer em `docker info`).
- Disco para as features do ShanghaiTech (bem menor que UCF-Crime, mas ainda
  assim vários GB).

## Passo a passo — PEL4VAD com ShanghaiTech e UCF-Crime

### 1. Configurar `DATA_ROOT`

```sh
cp .env.example .env
$EDITOR .env   # aponte DATA_ROOT para o diretório no host com os dados
```

`DATA_ROOT` é montado como `/data` (somente leitura) dentro do container do
PEL4VAD.

### 2. Baixar as features

**ShanghaiTech** baixa direto (Google Drive, automatizável):

```sh
DATA_ROOT=/caminho/para/dados ./scripts/download-features.sh pel4vad-sh
```

**UCF-Crime** está num OneDrive que bloqueia download via script (exige
browser) — passo a passo manual em
[docs/DATASETS.md §3.3](docs/DATASETS.md#33-pel4vad--ucf-crime-i3d-download-manual-via-onedrive).

Layout esperado em `DATA_ROOT` ao final:

```
$DATA_ROOT/
  shanghaitech/features/i3d/{train,test}
  ucf-crime/features/i3d/{train,test}
```

### 3. Confirmar que a GPU está visível

```sh
make GPU=1 check-gpu
```

Deve imprimir `cuda: True` para os três serviços (só nos importa o `pel4vad`
por enquanto).

### 4. Buildar e entrar no container

```sh
make GPU=1 up-pel4vad
```

Isso builda a imagem do PEL4VAD (na primeira vez) e abre um shell dentro do
container.

### 5. Treinar

Dentro do container, ajuste `PEL4VAD_FEAT_PREFIX` no `.env` do host conforme
o dataset (`/data/shanghaitech/features/i3d` ou `/data/ucf-crime/features/i3d`
— não é derivado de `--dataset`, ver [docs/DATASETS.md](docs/DATASETS.md)) e:

```sh
cd PEL4VAD
make train DATASET=sh    # ou DATASET=ucf
make infer DATASET=sh    # usa cfg.ckpt_path (checkpoints já vêm no repo)
```

### (Opcional) Validar sem dataset primeiro

Antes de baixar qualquer feature, dá para confirmar que a imagem, as
dependências e o loop de treino funcionam ponta a ponta com dados sintéticos:

```sh
make -f Makefile.smoke smoke-pel4vad
```

## Depois do PEL4VAD

Com o fluxo validado, AnomalyDetection (UCF-Crime) segue o mesmo padrão
(`make GPU=1 up-anomaly`, depois `make train`/`make roc` dentro do container)
— passo a passo completo em [docs/DATASETS.md](docs/DATASETS.md). RTFM e
AnomalyDetection+ShanghaiTech ficam fora deste primeiro momento — motivo em
[docs/DATASETS.md §7](docs/DATASETS.md#7-escopo-do-primeiro-momento-e-limitações-conhecidas).

## Estrutura do repositório

```
docker-compose.yml       # um serviço por modelo
docker-compose.gpu.yml   # reserva de GPU NVIDIA por serviço (make GPU=1 ...)
Makefile                 # build/up/check-gpu para os três modelos
Makefile.smoke           # smoke tests com features sintéticas (não versionado)
.env.example             # template de configuração local (DATA_ROOT etc.)
docs/DATASETS.md          # layout de dados, links de download, fluxo por modelo
models/
  RTFM/
  PEL4VAD/
  AnomalyDetectionCVPR2018-Pytorch/
```
