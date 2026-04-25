import os
import signal
import sys
import json
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))

if os.name == "nt":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    fallback_signal = getattr(signal, "SIGTERM", signal.SIGINT)
    for name in ("SIGUSR1", "SIGUSR2", "SIGCONT"):
        if not hasattr(signal, name):
            setattr(signal, name, fallback_signal)

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coeff * grad_output, None


def _gradient_reverse(x, coeff):
    if coeff <= 0:
        return x.detach()
    return _GradientReversal.apply(x, coeff)


def _scheduled_grl_lambda(module, cfg, base_coeff: float) -> float:
    warmup_frac = float(cfg.loss.get("grl_warmup_frac", 0.0))
    if base_coeff <= 0 or warmup_frac <= 0:
        return base_coeff

    trainer = getattr(module, "trainer", None)
    total_steps = getattr(trainer, "estimated_stepping_batches", None)
    if not total_steps:
        limit_batches = cfg.trainer.get("limit_train_batches", 1)
        if isinstance(limit_batches, int):
            total_steps = int(cfg.trainer.max_epochs) * max(1, int(limit_batches))
        else:
            total_steps = int(cfg.trainer.max_epochs)

    warmup_steps = max(1, int(float(total_steps) * warmup_frac))
    current_step = max(0, int(getattr(module, "global_step", 0)))
    return base_coeff * min(1.0, current_step / warmup_steps)


def _imagenet_normalization_stats(device, dtype):
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(stats["mean"], device=device, dtype=dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor(stats["std"], device=device, dtype=dtype).view(1, 1, 3, 1, 1)
    return mean, std


def _denormalize_pixels(pixels):
    mean, std = _imagenet_normalization_stats(pixels.device, pixels.dtype)
    return (pixels * std + mean).clamp(0.0, 1.0)


def _renormalize_pixels(pixels):
    mean, std = _imagenet_normalization_stats(pixels.device, pixels.dtype)
    return (pixels - mean) / std


def _expand_sequence_param(param, time_dim):
    param = param.squeeze(-1).squeeze(-1).squeeze(-1)
    if param.size(1) == 1 and time_dim > 1:
        param = param.expand(-1, time_dim)
    return param.unsqueeze(-1)


def _appearance_augment(pixels, aug_cfg):
    x = _denormalize_pixels(pixels)
    b, t, _, _, _ = x.shape
    shape = (b, 1 if getattr(aug_cfg, "consistent_across_time", False) else t, 1, 1, 1)
    zeros = torch.zeros(shape, device=x.device, dtype=x.dtype)

    brightness_delta = zeros
    if aug_cfg.brightness > 0:
        brightness_delta = 2 * torch.rand(shape, device=x.device, dtype=x.dtype) - 1
        x = x * (1 + brightness_delta * aug_cfg.brightness)

    contrast_delta = zeros
    if aug_cfg.contrast > 0:
        contrast_delta = 2 * torch.rand(shape, device=x.device, dtype=x.dtype) - 1
        contrast = 1 + contrast_delta * aug_cfg.contrast
        channel_mean = x.mean(dim=(3, 4), keepdim=True)
        x = (x - channel_mean) * contrast + channel_mean

    saturation_delta = zeros
    if aug_cfg.saturation > 0:
        saturation_delta = 2 * torch.rand(shape, device=x.device, dtype=x.dtype) - 1
        saturation = 1 + saturation_delta * aug_cfg.saturation
        gray = x.mean(dim=2, keepdim=True)
        x = gray + (x - gray) * saturation

    grayscale_mask = zeros
    if aug_cfg.grayscale_prob > 0:
        gray = x.mean(dim=2, keepdim=True).expand_as(x)
        grayscale_mask = (torch.rand(shape, device=x.device) < aug_cfg.grayscale_prob).to(x.dtype)
        mask = grayscale_mask.bool().expand_as(x)
        x = torch.where(mask, gray, x)

    noise_strength = zeros
    if aug_cfg.noise_std > 0:
        noise_strength = torch.rand(shape, device=x.device, dtype=x.dtype)
        x = x + torch.randn_like(x) * (aug_cfg.noise_std * noise_strength)

    nuisance_targets = torch.cat(
        [
            _expand_sequence_param(brightness_delta, t),
            _expand_sequence_param(contrast_delta, t),
            _expand_sequence_param(saturation_delta, t),
            _expand_sequence_param(grayscale_mask, t),
            _expand_sequence_param(noise_strength, t),
        ],
        dim=-1,
    )

    return _renormalize_pixels(x.clamp(0.0, 1.0)), nuisance_targets


def _appearance_stats_targets(pixels):
    x = _denormalize_pixels(pixels)
    channel_mean = x.mean(dim=(3, 4))
    channel_std = x.std(dim=(3, 4), unbiased=False)
    return torch.cat([channel_mean, channel_std], dim=-1)


def _cross_covariance_loss(x, y):
    x = x.reshape(-1, x.size(-1))
    y = y.reshape(-1, y.size(-1))
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    denom = max(x.size(0) - 1, 1)
    cov = (x.transpose(0, 1) @ y) / denom
    return cov.square().mean()


def _episode_split(dataset, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    num_episodes = len(dataset.lengths)
    if num_episodes < 2:
        raise ValueError("Episode-disjoint split requires at least two episodes")
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError(f"train_split must be between 0 and 1, got {train_fraction}")

    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_episodes, generator=generator).tolist()
    num_train = int(num_episodes * float(train_fraction))
    num_train = max(1, min(num_episodes - 1, num_train))
    train_episodes = sorted(int(idx) for idx in perm[:num_train])
    val_episodes = sorted(int(idx) for idx in perm[num_train:])
    return train_episodes, val_episodes


def _clip_indices_for_episodes(dataset, episode_indices: list[int]) -> list[int]:
    episode_set = set(int(idx) for idx in episode_indices)
    indices = [
        sample_idx
        for sample_idx, (episode_idx, _) in enumerate(dataset.clip_indices)
        if int(episode_idx) in episode_set
    ]
    if not indices:
        raise ValueError("Episode split produced an empty sample subset")
    return indices


@contextmanager
def _freeze_batchnorm_stats(module):
    batchnorm_states = []
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
            batchnorm_states.append((submodule, submodule.training))
            submodule.eval()
    try:
        yield
    finally:
        for submodule, was_training in batchnorm_states:
            submodule.train(was_training)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    loss = output["pred_loss"] + lambd * output["sigreg_loss"]

    aflwm_cfg = cfg.get("aflwm")
    if aflwm_cfg and aflwm_cfg.enabled:
        if "app_emb" not in output:
            raise RuntimeError("AF-LeWM requires app_emb from the model encode path")

        aug_a_pixels, nuisance_target_a = _appearance_augment(batch["pixels"], aflwm_cfg.augment)
        aug_b_pixels, nuisance_target_b = _appearance_augment(batch["pixels"], aflwm_cfg.augment)

        with _freeze_batchnorm_stats(self.model.projector):
            aug_a = self.model.project_features(self.model.encode_pixels(aug_a_pixels))
            aug_b = self.model.project_features(self.model.encode_pixels(aug_b_pixels))

        dyn_a = F.normalize(aug_a["emb"], dim=-1)
        dyn_b = F.normalize(aug_b["emb"], dim=-1)
        if aflwm_cfg.get("invariance", {}).get("stopgrad_target", False):
            dyn_b = dyn_b.detach()
        output["appearance_inv_loss"] = (dyn_a - dyn_b).pow(2).mean()
        output["appearance_indep_loss"] = _cross_covariance_loss(emb, output["app_emb"])

        stats_weight = float(aflwm_cfg.loss.get("stats_weight", 0.0))
        if getattr(self.model, "appearance_head", None) is not None and stats_weight > 0:
            with torch.no_grad():
                stats_target = _appearance_stats_targets(batch["pixels"])
            output["appearance_stats_loss"] = F.mse_loss(
                self.model.predict_appearance_stats(output["app_emb"]),
                stats_target,
            )

        nuisance_weight = float(aflwm_cfg.loss.get("nuisance_weight", 0.0))
        if getattr(self.model, "appearance_nuisance_head", None) is not None and nuisance_weight > 0:
            output["appearance_nuisance_loss"] = 0.5 * (
                F.mse_loss(self.model.predict_appearance_nuisance(aug_a["app_emb"]), nuisance_target_a)
                + F.mse_loss(self.model.predict_appearance_nuisance(aug_b["app_emb"]), nuisance_target_b)
            )

        dyn_nuisance_weight = float(aflwm_cfg.loss.get("dynamics_nuisance_weight", 0.0))
        if getattr(self.model, "dynamics_nuisance_head", None) is not None and dyn_nuisance_weight > 0:
            grl_lambda = _scheduled_grl_lambda(
                self,
                aflwm_cfg,
                float(aflwm_cfg.loss.get("grl_lambda", 1.0)),
            )
            output["grl_lambda_loss"] = torch.as_tensor(
                grl_lambda,
                device=emb.device,
                dtype=emb.dtype,
            )
            output["dynamics_nuisance_loss"] = 0.5 * (
                F.mse_loss(
                    self.model.predict_dynamics_nuisance(_gradient_reverse(aug_a["emb"], grl_lambda)),
                    nuisance_target_a,
                )
                + F.mse_loss(
                    self.model.predict_dynamics_nuisance(_gradient_reverse(aug_b["emb"], grl_lambda)),
                    nuisance_target_b,
                )
            )

        loss = (
            loss
            + aflwm_cfg.loss.invariance_weight * output["appearance_inv_loss"]
            + aflwm_cfg.loss.independence_weight * output["appearance_indep_loss"]
        )
        if "appearance_stats_loss" in output:
            loss = loss + stats_weight * output["appearance_stats_loss"]
        if "appearance_nuisance_loss" in output:
            loss = loss + nuisance_weight * output["appearance_nuisance_loss"]
        if "dynamics_nuisance_loss" in output:
            loss = loss + dyn_nuisance_weight * output["dynamics_nuisance_loss"]

    output["loss"] = loss

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(
        losses_dict,
        on_step=(stage in ("train", "fit")),
        on_epoch=True,
        sync_dist=True,
        batch_size=batch["pixels"].size(0),
    )
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm_pusht_official_budget")
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    train_episodes, val_episodes = _episode_split(dataset, cfg.train_split, int(cfg.seed))
    train_indices = _clip_indices_for_episodes(dataset, train_episodes)
    val_indices = _clip_indices_for_episodes(dataset, val_episodes)

    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col, episode_indices=train_episodes)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    loader_gen = torch.Generator().manual_seed(int(cfg.seed) + 1)
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=loader_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    appearance_projector = None
    appearance_head = None
    appearance_nuisance_head = None
    dynamics_nuisance_head = None
    aflwm_cfg = cfg.get("aflwm")
    if aflwm_cfg and aflwm_cfg.enabled:
        appearance_projector = MLP(
            input_dim=hidden_dim,
            output_dim=aflwm_cfg.appearance_dim,
            hidden_dim=aflwm_cfg.projector_hidden_dim,
            norm_fn=torch.nn.LayerNorm,
        )
        if float(aflwm_cfg.loss.get("stats_weight", 0.0)) > 0:
            appearance_head = MLP(
                input_dim=aflwm_cfg.appearance_dim,
                output_dim=6,
                hidden_dim=aflwm_cfg.stats_hidden_dim,
                norm_fn=torch.nn.LayerNorm,
            )
        if float(aflwm_cfg.loss.get("nuisance_weight", 0.0)) > 0:
            appearance_nuisance_head = MLP(
                input_dim=aflwm_cfg.appearance_dim,
                output_dim=5,
                hidden_dim=aflwm_cfg.nuisance_hidden_dim,
                norm_fn=torch.nn.LayerNorm,
            )
        if float(aflwm_cfg.loss.get("dynamics_nuisance_weight", 0.0)) > 0:
            dynamics_nuisance_head = MLP(
                input_dim=embed_dim,
                output_dim=5,
                hidden_dim=aflwm_cfg.nuisance_hidden_dim,
                norm_fn=torch.nn.LayerNorm,
            )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        appearance_projector=appearance_projector,
        appearance_head=appearance_head,
        appearance_nuisance_head=appearance_nuisance_head,
        dynamics_nuisance_head=dynamics_nuisance_head,
    )
    total_params = sum(p.numel() for p in world_model.parameters())

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    resume_enabled = bool(cfg.get("resume", False))
    weights_ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    final_object_path = (
        run_dir
        / f"{cfg.output_model_name}_epoch_{int(cfg.trainer.max_epochs)}_object.ckpt"
    )
    existing_artifacts = [
        path
        for path in [
            weights_ckpt_path,
            final_object_path,
            run_dir / "metrics" / "metrics.csv",
            run_dir / "train_metadata.json",
            run_dir / "split_metadata.json",
        ]
        if path.exists()
    ]
    existing_artifacts.extend(run_dir.glob(f"{cfg.output_model_name}_epoch_*_object.ckpt"))
    if not resume_enabled and existing_artifacts:
        raise RuntimeError(
            "Refusing to mix a fresh reliable run with existing artifacts. "
            f"Use a new output_model_name/subdir or set resume=True. Existing: {existing_artifacts}"
        )
    if resume_enabled and not weights_ckpt_path.exists():
        raise RuntimeError(f"resume=True but checkpoint does not exist: {weights_ckpt_path}")

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    else:
        logger = CSVLogger(save_dir=str(run_dir), name="metrics", version="")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    split_metadata = {
        "split": "episode_disjoint",
        "seed": int(cfg.seed),
        "train_split": float(cfg.train_split),
        "num_episodes": len(dataset.lengths),
        "train_episode_count": len(train_episodes),
        "val_episode_count": len(val_episodes),
        "train_sample_count": len(train_indices),
        "val_sample_count": len(val_indices),
        "normalizer_episode_scope": "train",
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
    }
    with open(run_dir / "split_metadata.json", "w") as f:
        json.dump(split_metadata, f, indent=2)

    callbacks = []
    if cfg.get("dump_object", True):
        object_epoch_interval = int(
            cfg.get("object_epoch_interval", int(cfg.trainer.max_epochs) + 1)
        )
        callbacks.append(
            ModelObjectCallBack(
                dirpath=run_dir,
                filename=cfg.output_model_name,
                epoch_interval=object_epoch_interval,
            )
        )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=resume_enabled,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=weights_ckpt_path if resume_enabled else None,
        seed=int(cfg.seed),
    )

    train_start_time = time.time()
    manager()
    train_elapsed_seconds = time.time() - train_start_time
    with open(run_dir / "train_metadata.json", "w") as f:
        json.dump(
            {
                "output_model_name": str(cfg.output_model_name),
                "seed": int(cfg.seed),
                "max_epochs": int(cfg.trainer.max_epochs),
                "limit_train_batches": int(cfg.trainer.limit_train_batches),
                "limit_val_batches": int(cfg.trainer.limit_val_batches),
                "batch_size": int(cfg.loader.batch_size),
                "model_params": int(total_params),
                "train_elapsed_seconds": train_elapsed_seconds,
            },
            f,
            indent=2,
        )
    if cfg.get("dump_object", True):
        if not final_object_path.exists():
            raise RuntimeError(
                f"Expected final object checkpoint was not written: {final_object_path}"
            )
    return


if __name__ == "__main__":
    run()
