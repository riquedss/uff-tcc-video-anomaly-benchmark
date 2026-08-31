"""Reaponta as .list do RTFM para o diretório local de features.

As lists commitadas no repo (`list/*.list`) guardam caminhos absolutos da
máquina dos autores originais (`/home/yu/yu_ssd/...`). Este script reescreve
apenas o diretório de cada linha, mantendo o nome do arquivo e -- crucialmente
-- a ordem original.

A ordem importa em dois lugares:
  - `dataset.py` separa anômalo de normal por índice fixo (`[:63]`/`[63:]` no
    shanghai, `[:810]`/`[810:]` no ucf), não pelo nome do arquivo;
  - a list de teste tem que casar linha a linha com o `gt-*.npy`.
Por isso não regeneramos as lists por glob do diretório: só trocamos o prefixo.
"""

import argparse
import os
import sys

DEFAULT_LISTS = {
    "shanghai": {
        "train": "list/shanghai-i3d-train-10crop.list",
        "test": "list/shanghai-i3d-test-10crop.list",
    },
    "ucf": {
        "train": "list/ucf-i3d.list",
        "test": "list/ucf-i3d-test.list",
    },
}


def rewrite(source_list: str, features_dir: str, out_list: str) -> int:
    with open(source_list) as fp:
        names = [os.path.basename(line.strip()) for line in fp if line.strip()]

    missing = [n for n in names if not os.path.exists(os.path.join(features_dir, n))]

    os.makedirs(os.path.dirname(out_list) or ".", exist_ok=True)
    with open(out_list, "w") as fp:
        for name in names:
            fp.write(os.path.join(features_dir, name) + "\n")

    print(f"{out_list}: {len(names)} linhas a partir de {source_list}")
    if missing:
        print(
            f"  AVISO: {len(missing)}/{len(names)} features não encontradas em "
            f"{features_dir}. Primeiras: {missing[:5]}",
            file=sys.stderr,
        )
    return len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_LISTS), required=True)
    parser.add_argument(
        "--train-features-dir",
        required=True,
        help="diretório com as features I3D 10-crop de treino (ex: /data/shanghaitech/features/i3d-10crop/train)",
    )
    parser.add_argument(
        "--test-features-dir",
        required=True,
        help="diretório com as features I3D 10-crop de teste",
    )
    parser.add_argument(
        "--out-dir",
        default="list/local",
        help="onde escrever as lists reapontadas (default: list/local)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="sai com erro se alguma feature listada não existir",
    )
    args = parser.parse_args()

    sources = DEFAULT_LISTS[args.dataset]
    missing = 0
    missing += rewrite(
        sources["train"],
        args.train_features_dir,
        os.path.join(args.out_dir, os.path.basename(sources["train"])),
    )
    missing += rewrite(
        sources["test"],
        args.test_features_dir,
        os.path.join(args.out_dir, os.path.basename(sources["test"])),
    )

    if missing and args.strict:
        raise SystemExit(f"{missing} features ausentes (--strict)")


if __name__ == "__main__":
    main()
