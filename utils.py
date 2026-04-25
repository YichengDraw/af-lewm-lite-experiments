import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def _episode_row_indices(dataset, episode_indices):
    rows = []
    for episode_idx in episode_indices:
        start = int(dataset.offsets[int(episode_idx)])
        end = start + int(dataset.lengths[int(episode_idx)])
        rows.extend(range(start, end))
    return rows


def get_column_normalizer(dataset, source: str, target: str, episode_indices=None, eps: float = 1e-6):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    if episode_indices is not None:
        col_data = col_data[_episode_row_indices(dataset, episode_indices)]
    data = torch.from_numpy(np.array(col_data)).float()
    if data.ndim == 1:
        data = data[~torch.isnan(data)]
    else:
        data = data[~torch.isnan(data).any(dim=1)]
    if data.numel() == 0:
        raise ValueError(f"No finite data available to normalize column '{source}'")
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone().clamp_min(eps)

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            is_interval_epoch = (trainer.current_epoch + 1) % self.epoch_interval == 0
            is_final_epoch = (trainer.current_epoch + 1) == trainer.max_epochs
            if is_interval_epoch or is_final_epoch:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(model, tmp_path)
            tmp_path.replace(path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"Error saving model object: {e}")
            raise
