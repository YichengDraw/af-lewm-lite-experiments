"""Download pretrained LeWM weights from HuggingFace and convert to object checkpoints."""
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))

if os.name == "nt":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import torch
import stable_pretraining as spt
from huggingface_hub import hf_hub_download
from jepa import JEPA
from module import ARPredictor, Embedder, MLP

# Environment configs: HF repo name -> (eval policy path, action_input_dim)
ENVS = {
    "pusht":    {"hf_repo": "quentinll/lewm-pusht",     "policy_dir": "pusht/lewm",    "action_dim": 10},
    "tworooms": {"hf_repo": "quentinll/lewm-tworooms",  "policy_dir": "tworoom/lewm",  "action_dim": 10},
    "cube":     {"hf_repo": "quentinll/lewm-cube",       "policy_dir": "cube/lewm",     "action_dim": 25},
    "reacher":  {"hf_repo": "quentinll/lewm-reacher",    "policy_dir": "reacher/lewm",  "action_dim": 10},
}


def build_model(action_dim: int, config: dict) -> JEPA:
    """Build a JEPA model from config."""
    enc_cfg = config["encoder"]
    pred_cfg = config["predictor"]

    encoder = spt.backbone.utils.vit_hf(
        enc_cfg["size"],
        patch_size=enc_cfg["patch_size"],
        image_size=enc_cfg["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = pred_cfg["input_dim"]

    predictor = ARPredictor(
        num_frames=pred_cfg["num_frames"],
        input_dim=embed_dim,
        hidden_dim=pred_cfg["hidden_dim"],
        output_dim=pred_cfg["output_dim"],
        depth=pred_cfg["depth"],
        heads=pred_cfg["heads"],
        mlp_dim=pred_cfg["mlp_dim"],
        dim_head=pred_cfg["dim_head"],
        dropout=pred_cfg["dropout"],
        emb_dropout=pred_cfg["emb_dropout"],
    )

    action_encoder = Embedder(input_dim=action_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=config["projector"]["input_dim"],
        output_dim=config["projector"]["output_dim"],
        hidden_dim=config["projector"]["hidden_dim"],
        norm_fn=torch.nn.BatchNorm1d,
    )

    pred_proj = MLP(
        input_dim=config["pred_proj"]["input_dim"],
        output_dim=config["pred_proj"]["output_dim"],
        hidden_dim=config["pred_proj"]["hidden_dim"],
        norm_fn=torch.nn.BatchNorm1d,
    )

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )


def main():
    cache_dir = Path(os.environ["STABLEWM_HOME"])

    for env_name, env_cfg in ENVS.items():
        print(f"\n{'='*60}")
        print(f"Processing {env_name}...")
        print(f"{'='*60}")

        out_dir = cache_dir / env_cfg["policy_dir"]
        out_path = out_dir / "lewm_object.ckpt"

        if out_path.exists():
            print(f"  SKIP - {out_path} already exists")
            continue

        # Download from HuggingFace
        print(f"  Downloading config.json from {env_cfg['hf_repo']}...")
        config_path = hf_hub_download(env_cfg["hf_repo"], "config.json")
        print(f"  Downloading weights.pt from {env_cfg['hf_repo']}...")
        weights_path = hf_hub_download(env_cfg["hf_repo"], "weights.pt")

        with open(config_path) as f:
            config = json.load(f)

        # Build model and load weights
        print(f"  Building model (action_dim={env_cfg['action_dim']})...")
        model = build_model(env_cfg["action_dim"], config)

        print(f"  Loading weights...")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # Save as object checkpoint
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Saving object checkpoint to {out_path}...")
        torch.save(model, out_path)

        # Verify
        loaded = torch.load(out_path, map_location="cpu", weights_only=False)
        assert hasattr(loaded, "get_cost"), f"Loaded model missing get_cost method!"
        print(f"  OK - Verified: model has get_cost method")

    print(f"\n{'='*60}")
    print("All checkpoints downloaded and converted!")
    print(f"{'='*60}")
    print("\nEval commands:")
    print("  python eval.py --config-name=pusht.yaml policy=pusht/lewm")
    print("  python eval.py --config-name=tworoom.yaml policy=tworoom/lewm")
    print("  python eval.py --config-name=cube.yaml policy=cube/lewm")
    print("  python eval.py --config-name=reacher.yaml policy=reacher/lewm")


if __name__ == "__main__":
    main()
