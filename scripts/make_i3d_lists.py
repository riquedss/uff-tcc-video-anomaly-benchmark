"""Gera as listas de vídeo (10 crops por linha) para RTFM e Sultani a partir das
listas do PEL4VAD, para que os três modelos usem as mesmas features I3D 1024-d
e o mesmo ground truth.

Uso: python3 scripts/make_i3d_lists.py
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEL4VAD_LIST = os.path.join(ROOT, "models", "PEL4VAD", "list")
OUT = {
    "rtfm": os.path.join(ROOT, "models", "RTFM", "list", "i3d"),
    "sultani": os.path.join(ROOT, "models", "AnomalyDetectionCVPR2018-Pytorch", "list_i3d"),
}
# Quantos vídeos anômalos abrem a lista de treino: o RTFM separa por índice.
N_ANOMALOUS = {"sh": 63, "ucf": 810}
CROP_RE = re.compile(r"^(.*?)(?:__\d|_\d)?\.npy$")


def read_videos(path):
    with open(path) as f:
        rows = [line.split() for line in f if line.strip()]
    assert len(rows) % 10 == 0, f"{path}: {len(rows)} linhas, não múltiplo de 10"
    videos = []
    for i in range(0, len(rows), 10):
        group = rows[i:i + 10]
        ids = {CROP_RE.match(os.path.basename(r[0])).group(1) for r in group}
        assert len(ids) == 1, f"{path}: crops de vídeos diferentes no grupo {i // 10}: {ids}"
        # SH traz o rótulo na lista; no UCF ele vem do nome ("Normal_Videos...").
        labels = {r[1] if len(r) > 1 else ("0" if "Normal" in r[0] else "1") for r in group}
        assert len(labels) == 1, f"{path}: rótulos diferentes no grupo {i // 10}"
        videos.append(([r[0] for r in group], labels.pop()))
    return videos


def write(path, videos, with_label):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for crops, label in videos:
            line = ",".join(crops)
            if with_label:
                line += " " + ("1" if float(label) > 0 else "0")
            f.write(line + "\n")


def main():
    for ds in ("sh", "ucf"):
        train = read_videos(os.path.join(PEL4VAD_LIST, ds, "train.list"))
        test = read_videos(os.path.join(PEL4VAD_LIST, ds, "test.list"))
        # Ordenação estável: anômalos primeiro, preservando a ordem relativa.
        train = sorted(train, key=lambda v: float(v[1]) == 0)
        n_abn = sum(float(v[1]) > 0 for v in train)
        assert n_abn == N_ANOMALOUS[ds], f"{ds}: {n_abn} anômalos, esperado {N_ANOMALOUS[ds]}"
        for model, out in OUT.items():
            write(os.path.join(out, f"{ds}-train.list"), train, with_label=(model == "sultani"))
            write(os.path.join(out, f"{ds}-test.list"), test, with_label=False)
        print(f"{ds}: treino {len(train)} vídeos ({n_abn} anômalos), teste {len(test)}")


if __name__ == "__main__":
    main()
