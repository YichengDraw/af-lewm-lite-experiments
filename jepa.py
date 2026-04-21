"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def select_goal_pixels(goal):
    """Return a single goal frame sequence with shape (B, 1, C, H, W)."""
    if goal.ndim == 4:
        return goal.unsqueeze(1)
    if goal.ndim == 5:
        return goal[:, -1:]
    if goal.ndim == 6:
        return goal[:, 0, -1:]
    raise ValueError(f"Unsupported goal tensor shape: {tuple(goal.shape)}")


def infer_history_size_from_predictor(predictor, default: int = 1) -> int:
    pos_embedding = getattr(predictor, "pos_embedding", None)
    if pos_embedding is None:
        return default
    return int(pos_embedding.shape[1])


class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        appearance_projector=None,
        appearance_head=None,
        appearance_nuisance_head=None,
        dynamics_nuisance_head=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.appearance_projector = appearance_projector
        self.appearance_head = appearance_head
        self.appearance_nuisance_head = appearance_nuisance_head
        self.dynamics_nuisance_head = dynamics_nuisance_head

    def encode_pixels(self, pixels):
        """Encode pixel observations into backbone features."""

        pixels = pixels.float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        return rearrange(pixels_emb, "(b t) d -> b t d", b=b)

    def project_features(self, encoder_emb):
        """Project backbone features into planning and appearance latents."""

        b = encoder_emb.size(0)
        flat_emb = rearrange(encoder_emb, "b t d -> (b t) d")
        dyn_emb = self.projector(flat_emb)

        output = {
            "encoder_emb": encoder_emb,
            "emb": rearrange(dyn_emb, "(b t) d -> b t d", b=b),
        }

        appearance_projector = getattr(self, "appearance_projector", None)
        if appearance_projector is not None:
            app_emb = appearance_projector(flat_emb)
            output["app_emb"] = rearrange(app_emb, "(b t) d -> b t d", b=b)

            appearance_head = getattr(self, "appearance_head", None)
            if appearance_head is not None:
                app_stats = appearance_head(app_emb)
                output["app_stats_pred"] = rearrange(app_stats, "(b t) d -> b t d", b=b)

        return output

    def predict_appearance_stats(self, app_emb):
        """Predict coarse image appearance statistics from the appearance latent."""

        appearance_head = getattr(self, "appearance_head", None)
        if appearance_head is None:
            raise RuntimeError("appearance_head is not defined on this JEPA instance")

        b = app_emb.size(0)
        flat_emb = rearrange(app_emb, "b t d -> (b t) d")
        pred = appearance_head(flat_emb)
        return rearrange(pred, "(b t) d -> b t d", b=b)

    def predict_appearance_nuisance(self, app_emb):
        """Predict sampled appearance nuisance parameters from the appearance latent."""

        nuisance_head = getattr(self, "appearance_nuisance_head", None)
        if nuisance_head is None:
            raise RuntimeError("appearance_nuisance_head is not defined on this JEPA instance")

        b = app_emb.size(0)
        flat_emb = rearrange(app_emb, "b t d -> (b t) d")
        pred = nuisance_head(flat_emb)
        return rearrange(pred, "(b t) d -> b t d", b=b)

    def predict_dynamics_nuisance(self, emb):
        """Predict nuisance parameters from the dynamics latent."""

        nuisance_head = getattr(self, "dynamics_nuisance_head", None)
        if nuisance_head is None:
            raise RuntimeError("dynamics_nuisance_head is not defined on this JEPA instance")

        b = emb.size(0)
        flat_emb = rearrange(emb, "b t d -> (b t) d")
        pred = nuisance_head(flat_emb)
        return rearrange(pred, "(b t) d -> b t d", b=b)

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        encoder_emb = self.encode_pixels(info["pixels"])
        info.update(self.project_features(encoder_emb))

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        if emb.shape[:2] != act_emb.shape[:2]:
            raise ValueError(
                f"Embedding/action context length mismatch: {emb.shape[:2]} vs {act_emb.shape[:2]}"
            )
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"

        B, S, T = action_sequence.shape[:3]
        pixels = info["pixels"]
        if pixels.ndim == 5:
            pixels = pixels.unsqueeze(1).expand(B, S, -1, -1, -1, -1)
        elif pixels.ndim == 6:
            if pixels.size(0) != B:
                raise ValueError(
                    f"Pixel batch size {pixels.size(0)} does not match action batch size {B}"
                )
            if pixels.size(1) == 1 and S > 1:
                pixels = pixels.expand(B, S, -1, -1, -1, -1)
            elif pixels.size(1) != S:
                raise ValueError(
                    f"Pixel sample count {pixels.size(1)} does not match action sample count {S}"
                )
        else:
            raise ValueError(f"Unsupported pixel tensor shape: {tuple(pixels.shape)}")

        if pixels.size(2) < history_size:
            raise ValueError(
                f"Need {history_size} history frames, got {pixels.size(2)}"
            )
        HS = history_size

        # copy and encode initial info dict
        init_info = {
            k: v[:, 0]
            for k, v in info.items()
            if torch.is_tensor(v) and k != "action"
        }
        init_info["pixels"] = pixels[:, 0]
        init_info = self.encode(init_info)
        emb = info["emb"] = init_info["emb"].unsqueeze(1).expand(B, S, -1, -1)

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        future_actions = rearrange(action_sequence, "b s ... -> (b s) ...")

        past_actions = info.get("action")
        if past_actions is None:
            past_actions = future_actions.new_zeros(
                future_actions.size(0), 0, future_actions.size(-1)
            )
        else:
            if past_actions.ndim == 3:
                past_actions = past_actions.unsqueeze(1).expand(B, S, -1, -1)
            elif past_actions.ndim == 4:
                if past_actions.size(1) == 1 and S > 1:
                    past_actions = past_actions.expand(B, S, -1, -1)
                elif past_actions.size(1) != S:
                    raise ValueError(
                        f"Past action sample count {past_actions.size(1)} does not match action sample count {S}"
                    )
            else:
                raise ValueError(
                    f"Unsupported past action tensor shape: {tuple(past_actions.shape)}"
                )
            past_actions = rearrange(past_actions, "b s ... -> (b s) ...")

        prefix_len = max(HS - 1, 0)
        if prefix_len > 0:
            if past_actions.size(1) < prefix_len:
                raise ValueError(
                    f"Need {prefix_len} past action blocks, got {past_actions.size(1)}"
                )
            past_actions = past_actions[:, -prefix_len:]
        elif past_actions.size(1) > 0:
            past_actions = past_actions[:, :0]

        # rollout predictor autoregressively for each future action block
        for t in range(T):
            if prefix_len > 0:
                action_context = torch.cat(
                    [past_actions, future_actions[:, t : t + 1]], dim=1
                )
            else:
                action_context = future_actions[:, t : t + 1]

            act_emb = self.action_encoder(action_context)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            if prefix_len > 0:
                past_actions = torch.cat(
                    [past_actions, future_actions[:, t : t + 1]], dim=1
                )[:, -prefix_len:]

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self._get_cost(info_dict, action_candidates)
        finally:
            self.train(was_training)

    def _get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)
        action_candidates = action_candidates.to(device)

        goal_pixels = select_goal_pixels(info_dict["goal"])
        goal = self.encode({"pixels": goal_pixels})

        info_dict["goal_emb"] = goal["emb"].unsqueeze(1).expand(
            -1,
            action_candidates.shape[1],
            *([-1] * (goal["emb"].ndim - 1)),
        )
        history_size = infer_history_size_from_predictor(self.predictor)
        info_dict = self.rollout(
            info_dict, action_candidates, history_size=history_size
        )

        cost = self.criterion(info_dict)

        return cost
