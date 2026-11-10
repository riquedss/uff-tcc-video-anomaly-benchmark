"""This module contains a training procedure for video anomaly detection."""

import argparse
from os import makedirs, path

import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

import random

import numpy as np

from features_loader import FeaturesLoader
from i3d_features import I3DFeaturesLoader, evaluate
from network.anomaly_detector_model import (
    AnomalyDetector,
    RegularizedLoss,
    custom_objective,
)
from network.TorchUtils import TorchModel, get_torch_device
from utils.callbacks import DefaultModelCallback, TensorBoardCallback
from utils.utils import register_logger
from green_metrics import GreenRun, size_mb


def get_args() -> argparse.Namespace:
    """Reads command line args and returns the parser object the represent the
    specified arguments."""
    parser = argparse.ArgumentParser(
        description="Video Anomaly Detection Training Parser"
    )

    # io
    parser.add_argument("--features_path", default="features", help="path to features")
    parser.add_argument(
        "--annotation_path",
        default="Train_Annotation.txt",
        help="path to train annotation",
    )
    parser.add_argument(
        "--log_file", type=str, default="log.log", help="set logging file."
    )
    parser.add_argument(
        "--exps_dir",
        type=str,
        default="exps",
        help="path to the directory where models and tensorboard would be saved.",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="load a model for resume training"
    )

    # optimization
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="epochs interval for saving the model checkpoints",
    )
    parser.add_argument("--lr_base", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--iterations_per_epoch",
        type=int,
        default=20000,
        help="number of training iterations",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of training epochs"
    )

    # Features I3D do PEL4VAD (ver i3d_features.py); sem --i3d_train_list,
    # usa o fluxo original com features C3D em texto.
    parser.add_argument("--i3d_train_list", type=str, default=None)
    parser.add_argument("--i3d_test_list", type=str, default=None)
    parser.add_argument("--i3d_gt", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None, help="rótulo do dataset no CSV")
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Register directories
    register_logger(log_file=args.log_file)
    makedirs(args.exps_dir, exist_ok=True)
    models_dir = path.join(args.exps_dir, "models")
    tb_dir = path.join(args.exps_dir, "tensorboard")
    makedirs(models_dir, exist_ok=True)
    makedirs(tb_dir, exist_ok=True)

    # Optimizations
    device = get_torch_device()
    cudnn.benchmark = True  # enable cudnn tune
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Data loader
    if args.i3d_train_list:
        train_loader = I3DFeaturesLoader(
            list_path=args.i3d_train_list,
            prefix=args.features_path,
            iterations=args.iterations_per_epoch,
        )
    else:
        train_loader = FeaturesLoader(
            features_path=args.features_path,
            annotation_path=args.annotation_path,
            iterations=args.iterations_per_epoch,
        )

    feature_dim = train_loader.get_feature_dim

    # Model
    if args.checkpoint is not None and path.exists(args.checkpoint):
        model = TorchModel.load_model(args.checkpoint)
        assert feature_dim == model.model.input_dim, (
            f"Dimensionality mismatch between input of the model ({model.input_dim}) and the loader ({feature_dim})"
        )
    else:
        network = AnomalyDetector(feature_dim)
        model = TorchModel(network)

    model = model.to(device).train()
    # Training parameters
    """
    In the original paper:
        lr = 0.01
        epsilon = 1e-8
    """
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr_base, eps=1e-8)

    criterion = RegularizedLoss(model.get_model(), custom_objective).to(device)

    # Callbacks
    tb_writer = SummaryWriter(log_dir=tb_dir)
    model.register_callback(DefaultModelCallback(visualization_dir=args.exps_dir))
    model.register_callback(TensorBoardCallback(tb_writer=tb_writer))

    # Training
    dataset_label = args.dataset_name or path.basename(path.normpath(args.features_path))
    if args.i3d_train_list:
        notes = f"features=i3d; seed={args.seed}; ckpt=last"
    else:
        notes = "AUC/FAR calculados à parte via generate_ROC.py (não há eval durante o treino)"
    with GreenRun(
        model="anomaly-detection",
        dataset=dataset_label,
        feature_dir=args.features_path,
        notes=notes,
    ) as run:
        model.fit(
            train_iter=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=args.epochs,
            network_model_path_base=models_dir,
            save_every=args.save_every,
        )
        params_m = sum(p.numel() for p in model.get_model().parameters()) / 1e6
        final_ckpt = path.join(models_dir, f"epoch_{args.epochs}.pt")
        metrics = {}
        if args.i3d_test_list:
            roc_auc, ap, far = evaluate(
                model.get_model(), args.i3d_test_list, args.features_path, args.i3d_gt, device
            )
            print(f"final: auc {roc_auc:.4f} ap {ap:.4f} far {far:.5f}")
            metrics = dict(auc=round(roc_auc, 4), ap=round(ap, 4), far=round(far, 5))
        run.set_metrics(params_m=round(params_m, 4), ckpt_mb=size_mb(final_ckpt), **metrics)
