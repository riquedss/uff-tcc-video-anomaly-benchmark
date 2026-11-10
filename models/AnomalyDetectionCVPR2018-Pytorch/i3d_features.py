"""Features I3D 1024-d do PEL4VAD (um arquivo .npy por crop) no formato do
Sultani: média dos 10 crops e 32 segmentos por vídeo.

As listas (list_i3d/) têm um vídeo por linha, com os 10 caminhos separados por
vírgula; as de treino trazem o rótulo (1 = anômalo) na segunda coluna. Gerá-las
com scripts/make_i3d_lists.py na raiz do projeto.
"""

import os

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve

N_SEGMENTS = 32
FRAMES_PER_SNIPPET = 16


def read_list(path):
    with open(path) as f:
        rows = [line.split() for line in f if line.strip()]
    return [(r[0].split(","), int(r[1]) if len(r) > 1 else None) for r in rows]


def load_snippets(prefix, crops):
    """[T, F]: média das features dos 10 crops."""
    arrays = [np.load(os.path.join(prefix, c)) for c in crops]
    t = min(len(a) for a in arrays)
    return np.mean([a[:t] for a in arrays], axis=0).astype(np.float32)


def segment_bounds(n_snippets):
    return np.linspace(0, n_snippets, N_SEGMENTS + 1, dtype=int)


def to_segments(snippets):
    """[32, F]: média dos snippets de cada segmento, como nas features C3D originais."""
    r = segment_bounds(len(snippets))
    out = np.zeros((N_SEGMENTS, snippets.shape[1]), dtype=np.float32)
    for i in range(N_SEGMENTS):
        out[i] = snippets[r[i]:r[i + 1]].mean(0) if r[i] != r[i + 1] else snippets[min(r[i], len(snippets) - 1)]
    return out


class I3DFeaturesLoader:
    """Mesma interface do FeaturesLoader: cada item é um bucket de 30 vídeos
    normais e 30 anômalos sorteados com np.random."""

    def __init__(self, list_path, prefix, bucket_size=30, iterations=20000):
        videos = read_list(list_path)
        self._normal = [c for c, y in videos if y == 0]
        self._anomaly = [c for c, y in videos if y == 1]
        self._prefix = prefix
        self._bucket_size = bucket_size
        self._iterations = iterations
        self._cache = {}
        self._i = 0

    def __len__(self):
        return self._iterations

    def _segments(self, crops):
        key = crops[0]
        if key not in self._cache:
            self._cache[key] = torch.from_numpy(to_segments(load_snippets(self._prefix, crops)))
        return self._cache[key]

    def __getitem__(self, index):
        if self._i == len(self):
            self._i = 0
            raise StopIteration
        normal = np.random.choice(len(self._normal), size=self._bucket_size)
        anomaly = np.random.choice(len(self._anomaly), size=self._bucket_size)
        feats = [self._segments(self._normal[i]) for i in normal] + \
                [self._segments(self._anomaly[i]) for i in anomaly]
        labels = torch.cat([torch.zeros(self._bucket_size), torch.ones(self._bucket_size)])
        self._i += 1
        return torch.stack(feats), labels

    @property
    def get_feature_dim(self):
        crops = self._normal[0]
        return int(np.load(os.path.join(self._prefix, crops[0])).shape[-1])


def evaluate(model, list_path, prefix, gt_path, device, threshold=0.5):
    """AUC, AP e FAR no nível de quadro, no mesmo protocolo do PEL4VAD: o escore
    de cada segmento vale para os seus snippets, cada snippet vale 16 quadros, e
    tudo é comparado com o ground truth concatenado (list/<ds>/<ds>-gt.npy)."""
    gt = np.load(gt_path)
    preds, normal_preds = [], []
    offset = 0
    model.eval()
    with torch.no_grad():
        for crops, _ in read_list(list_path):
            snippets = load_snippets(prefix, crops)
            seg = torch.from_numpy(to_segments(snippets)).unsqueeze(0).to(device)
            seg_scores = model(seg).squeeze().cpu().numpy()
            r = segment_bounds(len(snippets))
            snip_scores = np.zeros(len(snippets), dtype=np.float32)
            for i in range(N_SEGMENTS):
                snip_scores[r[i]:r[i + 1]] = seg_scores[i]
            frame_scores = np.repeat(snip_scores, FRAMES_PER_SNIPPET)
            video_gt = gt[offset:offset + len(frame_scores)]
            if video_gt.sum() == 0:
                normal_preds.append(frame_scores)
            preds.append(frame_scores)
            offset += len(frame_scores)
    assert offset == len(gt), f"ground truth com {len(gt)} quadros, predições com {offset}"
    preds = np.concatenate(preds)
    fpr, tpr, _ = roc_curve(gt, preds)
    precision, recall, _ = precision_recall_curve(gt, preds)
    normal = np.concatenate(normal_preds) if normal_preds else np.zeros(0)
    far = float((normal >= threshold).mean()) if len(normal) else 0.0
    return float(auc(fpr, tpr)), float(auc(recall, precision)), far
