"""Gera as figuras do TCC a partir de results/remote (CSV de custo e logs de treino).

Uso: python3 scripts/plot_resultados.py [--results results/remote] [--out figuras]
"""
import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

MODELOS = {"anomaly-detection": "Sultani", "pel4vad": "PEL4VAD", "rtfm": "RTFM"}
COR_MODELO = {"Sultani": SERIES[1], "PEL4VAD": SERIES[0], "RTFM": SERIES[2]}
DATASETS = {"c3d": "UCF-Crime", "ucf-crime": "UCF-Crime", "ucf": "UCF-Crime",
            "shanghaiTech": "ShanghaiTech", "shanghaitech": "ShanghaiTech", "shanghai": "ShanghaiTech"}
MARCADOR = {"UCF-Crime": "o", "ShanghaiTech": "s"}
AUC_ARTIGO = {("PEL4VAD", "ShanghaiTech"): 0.9814, ("PEL4VAD", "UCF-Crime"): 0.8676}

EPOCA_RE = re.compile(r"\[Epoch:(\d+)/\d+\].*AUC:([\d.]+)")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.edgecolor": TEXT_2,
    "axes.labelcolor": TEXT,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


def br(x, casas=4):
    return f"{x:.{casas}f}".replace(".", ",")


def eixo_br(ax, casas=2, eixo="y"):
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: br(v, casas))
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(fmt)


def fnum(v):
    return float(v.replace(",", ".")) if v not in ("", None) else None


def ler_treinos(results):
    """Uma entrada por treino: custo da linha de treino + AUC do último checkpoint."""
    with open(os.path.join(results, "green_metrics.csv"), newline="") as f:
        linhas = list(csv.DictReader(f))

    treinos = {}
    for r in linhas:
        modelo = MODELOS.get(r["model"], r["model"])
        dataset = DATASETS.get(r["dataset"], r["dataset"])
        chave = (modelo, dataset)
        if r["energy_kwh"]:
            treinos[chave] = {
                "modelo": modelo, "dataset": dataset,
                "kwh": fnum(r["energy_kwh"]), "gpu": fnum(r["gpu_energy_kwh"]),
                "cpu": fnum(r["cpu_energy_kwh"]), "ram": fnum(r["ram_energy_kwh"]),
                "min": fnum(r["duration_s"]) / 60, "auc": fnum(r["auc"]),
            }
        elif "epoch_2.pt" in r["notes"] and chave in treinos:
            treinos[chave]["auc"] = fnum(r["auc"])

    for log, chave in curvas_por_log(results):
        if chave in treinos:
            treinos[chave]["auc"] = log[-1][1]
    return list(treinos.values())


def curvas_por_log(results):
    for path in sorted(glob.glob(os.path.join(results, "pel4vad_*_train*.log"))):
        dataset = "ShanghaiTech" if "_sh_" in os.path.basename(path) else "UCF-Crime"
        pontos = []
        with open(path) as f:
            for linha in f:
                m = EPOCA_RE.search(linha)
                if m:
                    pontos.append((int(m.group(1)), float(m.group(2))))
        if pontos:
            yield pontos, ("PEL4VAD", dataset)


def fig_desempenho_custo(treinos, out):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for t in treinos:
        ax.scatter(t["kwh"], t["auc"], s=70, marker=MARCADOR[t["dataset"]],
                   color=COR_MODELO[t["modelo"]], edgecolor="white", linewidth=1.5, zorder=3)
        rotulo = f'{t["modelo"]} · {t["dataset"]}' + (" (C3D)" if t["modelo"] == "Sultani" else "")
        ax.annotate(rotulo, (t["kwh"], t["auc"]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=TEXT)
    ax.set_xscale("log")
    ax.set_xlabel("Energia total do treino (kWh, escala log)")
    ax.set_ylabel("AUC (último checkpoint)")
    ax.grid(True, which="major")
    ax.set_xlim(right=ax.get_xlim()[1] * 4)
    eixo_br(ax, 2)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 2 if v < 1 else 0)))

    from matplotlib.lines import Line2D
    leg_modelo = [Line2D([], [], marker="o", ls="", color=COR_MODELO[m], label=m)
                  for m in sorted({t["modelo"] for t in treinos})]
    leg_dataset = [Line2D([], [], marker=MARCADOR[d], ls="", color=TEXT_2, label=d)
                   for d in sorted({t["dataset"] for t in treinos})]
    ax.legend(handles=leg_modelo + leg_dataset, loc="lower right", fontsize=9)
    fig.savefig(os.path.join(out, "desempenho_custo.pdf"))
    plt.close(fig)


def fig_auc_por_epoca(results, out):
    curvas = sorted(curvas_por_log(results), key=lambda c: c[1][1], reverse=True)
    fig, axes = plt.subplots(1, len(curvas), figsize=(3.2 * len(curvas), 3), squeeze=False)
    for ax, (pontos, chave) in zip(axes[0], curvas):
        ep, auc = zip(*pontos)
        ax.plot(ep, auc, color=SERIES[0], linewidth=2)
        i_best = max(range(len(auc)), key=auc.__getitem__)
        for i in (i_best, len(auc) - 1):
            ax.plot(ep[i], auc[i], "o", ms=7, color=SERIES[0], mec="white", mew=1.5, zorder=3)
        ax.text(0.98, 0.04, f"melhor: {br(auc[i_best])} (época {ep[i_best]})\núltima: {br(auc[-1])}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=TEXT,
                bbox=dict(facecolor="white", edgecolor="none", pad=2))
        if chave in AUC_ARTIGO:
            ref = AUC_ARTIGO[chave]
            ax.axhline(ref, color=TEXT_2, linewidth=1, linestyle="--")
            ax.annotate(f"artigo: {br(ref)}", (ep[-1], ref), xytext=(0, 3), ha="right",
                        textcoords="offset points", fontsize=8, color=TEXT_2)
            ax.set_ylim(top=ref + (ref - min(auc)) * 0.12)
        ax.set_title(f"{chave[0]} · {chave[1]}", fontsize=10, color=TEXT)
        ax.set_xlabel("Época")
        ax.grid(True, axis="y")
        eixo_br(ax, 3)
    axes[0][0].set_ylabel("AUC no conjunto de teste")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "auc_por_epoca.pdf"))
    plt.close(fig)


def fig_energia_componentes(treinos, out):
    treinos = sorted(treinos, key=lambda t: t["kwh"])
    fig, ax = plt.subplots(figsize=(6, 0.55 * len(treinos) + 1.2))
    nomes = [f'{t["modelo"]} · {t["dataset"]}' for t in treinos]
    esquerda = [0.0] * len(treinos)
    for comp, rotulo, cor in [("gpu", "GPU", SERIES[0]), ("cpu", "CPU", SERIES[1]), ("ram", "RAM", SERIES[2])]:
        frac = [100 * t[comp] / (t["gpu"] + t["cpu"] + t["ram"]) for t in treinos]
        ax.barh(nomes, frac, left=esquerda, color=cor, edgecolor="white", linewidth=2,
                height=0.6, label=rotulo)
        esquerda = [a + b for a, b in zip(esquerda, frac)]
    for i, t in enumerate(treinos):
        ax.annotate(f'{br(t["kwh"], 3)} kWh · {t["min"]:.0f} min', (100, i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Participação na energia do treino (%)")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(os.path.join(out, "energia_componentes.pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/remote")
    p.add_argument("--out", default="figuras")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    treinos = ler_treinos(a.results)
    fig_desempenho_custo(treinos, a.out)
    fig_auc_por_epoca(a.results, a.out)
    fig_energia_componentes(treinos, a.out)
    for t in treinos:
        print(t)


if __name__ == "__main__":
    main()
