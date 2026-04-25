from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

from train import (
    _appearance_augment,
    _clip_indices_for_episodes,
    _episode_split,
)
from utils import get_column_normalizer, get_img_preprocessor


STABLEWM_HOME = Path(os.environ["STABLEWM_HOME"]).expanduser()


def run_dir_from_policy(policy: str) -> Path:
    return STABLEWM_HOME / Path(policy).parent


def load_train_config(run_dir: Path):
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing train config for diagnostics: {config_path}")
    return OmegaConf.load(config_path)


def build_dataset(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    split_path = Path(swm.data.utils.get_cache_dir(), cfg.subdir) / "split_metadata.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
        train_episodes = [int(ep) for ep in split["train_episodes"]]
        val_episodes = [int(ep) for ep in split["val_episodes"]]
    else:
        train_episodes, val_episodes = _episode_split(dataset, cfg.train_split, int(cfg.seed))

    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col, episode_indices=train_episodes))
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(*transforms)
    return torch.utils.data.Subset(dataset, _clip_indices_for_episodes(dataset, val_episodes))


def flatten_latent(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().reshape(-1, value.size(-1))


def standard_cross_cov(x: torch.Tensor, y: torch.Tensor) -> float:
    if x is None or y is None:
        return float("nan")
    x = flatten_latent(x)
    y = flatten_latent(y)
    n = min(x.size(0), y.size(0))
    x = x[:n]
    y = y[:n]
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, unbiased=False, keepdim=True) + 1e-6)
    y = (y - y.mean(0, keepdim=True)) / (y.std(0, unbiased=False, keepdim=True) + 1e-6)
    cov = x.transpose(0, 1) @ y / max(n - 1, 1)
    return float(cov.square().mean().item())


def ridge_probe_mse(features: torch.Tensor | None, targets: torch.Tensor | None, ridge: float = 1e-3) -> float:
    if features is None or targets is None:
        return float("nan")
    x = flatten_latent(features).cpu()
    y = flatten_latent(targets).cpu()
    n = min(x.size(0), y.size(0))
    x = x[:n]
    y = y[:n]
    if n < 8:
        return float("nan")

    x = torch.cat([x, torch.ones(n, 1)], dim=1)
    split = max(1, int(0.8 * n))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    if x_test.numel() == 0:
        return float("nan")

    eye = torch.eye(x_train.size(1), dtype=x_train.dtype)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(
        x_train.transpose(0, 1) @ x_train + ridge * eye,
        x_train.transpose(0, 1) @ y_train,
    )
    pred = x_test @ weights
    return float(F.mse_loss(pred, y_test).item())


def clean_json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    return value


@torch.inference_mode()
def collect_metrics(model, loader, cfg, device: str, num_batches: int) -> dict[str, float]:
    model = model.to(device).eval()
    has_af = bool(cfg.get("aflwm") and cfg.aflwm.get("enabled", False))
    if not has_af or getattr(model, "appearance_projector", None) is None:
        return {
            "has_app_emb": False,
            "emb_aug_sensitivity": float("nan"),
            "app_aug_sensitivity": float("nan"),
            "emb_app_cross_cov": float("nan"),
            "emb_nuisance_probe_mse": float("nan"),
            "app_nuisance_probe_mse": float("nan"),
            "appearance_nuisance_head_mse": float("nan"),
            "dynamics_nuisance_head_mse": float("nan"),
        }

    emb_a = []
    emb_b = []
    app_a = []
    app_b = []
    clean_emb = []
    clean_app = []
    nuisance = []
    app_head_mse = []
    dyn_head_mse = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        clean = model.encode(batch.copy())
        aug_a_pixels, nuisance_a = _appearance_augment(batch["pixels"], cfg.aflwm.augment)
        aug_b_pixels, _ = _appearance_augment(batch["pixels"], cfg.aflwm.augment)
        aug_a = model.project_features(model.encode_pixels(aug_a_pixels))
        aug_b = model.project_features(model.encode_pixels(aug_b_pixels))

        clean_emb.append(clean["emb"].detach().cpu())
        clean_app.append(clean["app_emb"].detach().cpu())
        emb_a.append(aug_a["emb"].detach().cpu())
        emb_b.append(aug_b["emb"].detach().cpu())
        app_a.append(aug_a["app_emb"].detach().cpu())
        app_b.append(aug_b["app_emb"].detach().cpu())
        nuisance.append(nuisance_a.detach().cpu())

        if getattr(model, "appearance_nuisance_head", None) is not None:
            app_head_mse.append(
                F.mse_loss(model.predict_appearance_nuisance(aug_a["app_emb"]), nuisance_a).detach().cpu()
            )
        if getattr(model, "dynamics_nuisance_head", None) is not None:
            dyn_head_mse.append(
                F.mse_loss(model.predict_dynamics_nuisance(aug_a["emb"]), nuisance_a).detach().cpu()
            )

    emb_a_t = torch.cat(emb_a)
    emb_b_t = torch.cat(emb_b)
    app_a_t = torch.cat(app_a)
    app_b_t = torch.cat(app_b)
    clean_emb_t = torch.cat(clean_emb)
    clean_app_t = torch.cat(clean_app)
    nuisance_t = torch.cat(nuisance)

    return {
        "has_app_emb": True,
        "emb_aug_sensitivity": float(
            (F.normalize(emb_a_t.float(), dim=-1) - F.normalize(emb_b_t.float(), dim=-1)).pow(2).mean().sqrt().item()
        ),
        "app_aug_sensitivity": float(
            (F.normalize(app_a_t.float(), dim=-1) - F.normalize(app_b_t.float(), dim=-1)).pow(2).mean().sqrt().item()
        ),
        "emb_app_cross_cov": standard_cross_cov(clean_emb_t, clean_app_t),
        "emb_nuisance_probe_mse": ridge_probe_mse(emb_a_t, nuisance_t),
        "app_nuisance_probe_mse": ridge_probe_mse(app_a_t, nuisance_t),
        "appearance_nuisance_head_mse": float(torch.stack(app_head_mse).mean().item()) if app_head_mse else float("nan"),
        "dynamics_nuisance_head_mse": float(torch.stack(dyn_head_mse).mean().item()) if dyn_head_mse else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose AF-LeWM latent factorization.")
    parser.add_argument("--policy", required=True, help="Policy path without _object.ckpt suffix")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = run_dir_from_policy(args.policy)
    cfg = load_train_config(run_dir)
    dataset = build_dataset(cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    model = swm.policy.AutoCostModel(args.policy)
    metrics = collect_metrics(model, loader, cfg, args.device, args.num_batches)
    payload = {
        "policy": args.policy,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "metrics": metrics,
    }

    out_path = Path(args.output) if args.output else run_dir / "latent_diagnostics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = clean_json_value(payload)
    out_path.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
