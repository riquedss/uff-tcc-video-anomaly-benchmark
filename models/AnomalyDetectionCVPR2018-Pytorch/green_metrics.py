"""Wrapper comum de métricas de custo computacional ("IA Verde") para os 3
modelos do TCC: tempo de treino, energia, emissões estimadas e tamanho em
disco (features/checkpoint). Mede o que nenhum dos papers da literatura
reporta, para permitir comparação de eficiência entre os modelos.

Mesmo arquivo duplicado em models/{AnomalyDetectionCVPR2018-Pytorch,PEL4VAD,
RTFM} -- cada build do Docker tem contexto isolado nesse diretório, então não
dá pra compartilhar via import entre os 3 sem mudar a estrutura do repo.

Escreve uma linha por execução em um CSV único, compartilhado pelos 3
containers via bind mount (ver docker-compose.yml: ./results:/results), para
que a comparação final não precise de merge manual.

Uso:
    with GreenRun(model="pel4vad", dataset=cfg.dataset, feature_dir=cfg.feat_prefix) as run:
        best_auc, far = train(...)
        run.set_metrics(auc=best_auc, far=far, params_m=param / 1e6,
                         ckpt_mb=size_mb(ckpt_path))
"""

import csv
import os
import threading
import time
from pathlib import Path

import codecarbon.core.gpu
import pynvml
from codecarbon import OfflineEmissionsTracker

CSV_PATH = os.environ.get("GREEN_METRICS_CSV", "/results/green_metrics.csv")
FIELDS = [
    "timestamp", "model", "dataset", "duration_s", "energy_kwh",
    "emissions_kg", "gpu_energy_kwh", "cpu_energy_kwh", "ram_energy_kwh",
    "params_m", "feature_dir_mb", "ckpt_mb", "auc", "ap", "far", "notes",
]


def size_mb(path):
    """Tamanho em MB de um arquivo ou diretório (soma recursiva). None se
    não existir -- deixa a coluna em branco no CSV em vez de quebrar o run."""
    if not path or not os.path.exists(path):
        return None
    if os.path.isfile(path):
        return round(os.path.getsize(path) / (1024 ** 2), 2)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return round(total / (1024 ** 2), 2)


def _gpus_without_energy_counter():
    """Handles NVML das GPUs que não expõem o contador de energia acumulada
    (nvmlDeviceGetTotalEnergyConsumption, só existe da Volta em diante).
    Lista vazia se não há GPU NVIDIA ou se o contador funciona -- nesses
    casos o codecarbon mede sozinho."""
    try:
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                   for i in range(pynvml.nvmlDeviceGetCount())]
    except pynvml.NVMLError:
        return []
    try:
        for h in handles:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        return []
    except pynvml.NVMLError:
        return handles


class _GpuPowerSampler(threading.Thread):
    """Integra no tempo (regra do trapézio) a potência instantânea das GPUs,
    lida via nvmlDeviceGetPowerUsage. Substitui a medição de GPU do
    codecarbon em placas sem contador de energia (ex.: GTX TITAN X, Maxwell,
    da máquina da facu): lá o codecarbon falha no __init__ e zera tudo."""

    def __init__(self, handles, interval_s=1.0):
        super().__init__(daemon=True)
        self.handles = handles
        self.interval_s = interval_s
        self.energy_j = 0.0
        self._stop_evt = threading.Event()

    def _power_w(self):
        return sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in self.handles) / 1000.0

    def run(self):
        last_t, last_p = time.time(), self._power_w()
        while True:
            stopping = self._stop_evt.wait(self.interval_s)
            t, p = time.time(), self._power_w()
            self.energy_j += (p + last_p) / 2 * (t - last_t)
            last_t, last_p = t, p
            if stopping:
                return

    def stop_kwh(self):
        self._stop_evt.set()
        self.join()
        return self.energy_j / 3.6e6


class GreenRun:
    def __init__(self, model, dataset, feature_dir=None, notes=""):
        self.model = model
        self.dataset = dataset
        self.feature_dir = feature_dir
        self.notes = notes
        self.metrics = {}
        self.gpu_sampler = None
        handles = _gpus_without_energy_counter()
        if handles:
            # Esconde a GPU do codecarbon (ele chama gpu.is_gpu_details_available
            # pelo módulo no __init__) e mede via _GpuPowerSampler.
            codecarbon.core.gpu.is_gpu_details_available = lambda: False
            self.gpu_sampler = _GpuPowerSampler(handles)
            self.notes = " ".join(filter(None, [self.notes, "gpu_energy via amostragem de potencia NVML"]))
        # Offline: não depende de API externa p/ fator de emissão (máquina
        # de treino pode estar sem internet liberada) e fica reprodutível.
        self.tracker = OfflineEmissionsTracker(
            country_iso_code="BRA",
            log_level="error",
            save_to_file=False,
        )

    def set_metrics(self, **kwargs):
        self.metrics.update(kwargs)

    def __enter__(self):
        self._t0 = time.time()
        self.tracker.start()
        if self.gpu_sampler:
            self.gpu_sampler.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        emissions_kg = self.tracker.stop() or 0.0
        duration_s = time.time() - self._t0
        # stop() do codecarbon é decorado com @suppress(Exception): se algo
        # falhar lá dentro ele retorna None sem setar final_emissions_data.
        # Sem o getattr, esse AttributeError mascarava a exceção original
        # do treino e o CSV ficava sem a linha.
        data = getattr(self.tracker, "final_emissions_data", None)
        energy_kwh = getattr(data, "energy_consumed", 0.0) or 0.0
        gpu_energy_kwh = getattr(data, "gpu_energy", 0.0) or 0.0
        if self.gpu_sampler:
            sampled_kwh = self.gpu_sampler.stop_kwh()
            # Mesmo fator de emissão (kg/kWh) que o codecarbon usou p/ CPU+RAM.
            if energy_kwh:
                emissions_kg += sampled_kwh * emissions_kg / energy_kwh
            energy_kwh += sampled_kwh
            gpu_energy_kwh += sampled_kwh

        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": self.model,
            "dataset": self.dataset,
            "duration_s": round(duration_s, 1),
            "energy_kwh": round(energy_kwh, 6),
            "emissions_kg": round(emissions_kg, 6),
            "gpu_energy_kwh": round(gpu_energy_kwh, 6),
            "cpu_energy_kwh": round(getattr(data, "cpu_energy", 0.0) or 0.0, 6),
            "ram_energy_kwh": round(getattr(data, "ram_energy", 0.0) or 0.0, 6),
            "feature_dir_mb": size_mb(self.feature_dir),
            "notes": self.notes if exc_type is None else "FALHOU: " + " ".join(str(exc).split()),
        }
        row.update(self.metrics)

        out_dir = os.path.dirname(CSV_PATH) or "."
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in FIELDS})

        return False
