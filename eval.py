import os
import sys
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

if os.name != "nt":
    os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))

if os.name == "nt":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import time

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def infer_model_history_size(model) -> int:
    predictor = getattr(model, "predictor", None)
    pos_embedding = getattr(predictor, "pos_embedding", None)
    if pos_embedding is None:
        return 1
    return int(pos_embedding.shape[1])


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def policy_run_dir(policy: str) -> Path:
    return Path(swm.data.utils.get_cache_dir(), policy).parent


def episode_row_indices(dataset, episode_indices):
    rows = []
    for episode_idx in episode_indices:
        start = int(dataset.offsets[int(episode_idx)])
        end = start + int(dataset.lengths[int(episode_idx)])
        rows.extend(range(start, end))
    return rows


def resolve_normalizer_scope(dataset, policy: str):
    if policy == "random":
        return None, {
            "normalizer_scope": "full_dataset_random_policy",
            "split_metadata_path": None,
        }

    split_path = policy_run_dir(policy) / "split_metadata.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split metadata for train-split normalizers: {split_path}"
        )

    split_metadata = json.loads(split_path.read_text())
    train_episodes = split_metadata.get("train_episodes")
    if not train_episodes:
        raise ValueError(f"No train_episodes found in {split_path}")
    if split_metadata.get("normalizer_episode_scope") not in (None, "train"):
        raise ValueError(
            f"Unsupported normalizer scope in {split_path}: "
            f"{split_metadata.get('normalizer_episode_scope')}"
        )

    train_episodes = [int(ep) for ep in train_episodes]
    max_episode = len(dataset.lengths) - 1
    invalid = [ep for ep in train_episodes if ep < 0 or ep > max_episode]
    if invalid:
        raise ValueError(
            f"Split metadata references invalid episode ids: {invalid[:5]}"
        )

    return train_episodes, {
        "normalizer_scope": "train_episodes_from_split_metadata",
        "split_metadata_path": str(split_path),
        "train_episode_count": len(train_episodes),
    }


class ColumnStandardizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def transform(self, x):
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x):
        return (np.asarray(x, dtype=np.float32) * self.std + self.mean).astype(np.float32)


def fit_column_processor(dataset, col: str, episode_indices=None, eps: float = 1e-6):
    col_data = dataset.get_col_data(col)
    if episode_indices is not None:
        col_data = col_data[episode_row_indices(dataset, episode_indices)]
    if col_data.ndim == 1:
        col_data = col_data[~np.isnan(col_data)]
    else:
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
    if len(col_data) == 0:
        raise ValueError(f"No finite rows available to fit scaler for '{col}'")
    data = torch.from_numpy(np.asarray(col_data)).float()
    mean = data.mean(0, keepdim=True).numpy()
    std = data.std(0, keepdim=True).clamp_min(eps).numpy()
    return ColumnStandardizer(mean, std)


def build_processors(cfg, dataset, episode_indices=None):
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = fit_column_processor(dataset, col, episode_indices)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def build_planning_context(
    info_dict: dict,
    *,
    history_len: int,
    action_block: int,
    block_action_dim: int,
) -> dict:
    pixels = info_dict["pixels"]
    raw_actions = info_dict["action"]
    context_span = history_len * action_block

    if pixels.size(1) < context_span:
        raise ValueError(
            f"Need {context_span} raw history steps, got {pixels.size(1)}"
        )
    if raw_actions.size(1) < context_span:
        raise ValueError(
            f"Need {context_span} raw action steps, got {raw_actions.size(1)}"
        )

    pixels = pixels[:, -context_span:]
    raw_actions = raw_actions[:, -context_span:]
    select_idx = torch.arange(
        action_block - 1,
        context_span,
        action_block,
        device=pixels.device,
    )

    planner_info = {
        "pixels": pixels.index_select(1, select_idx),
    }

    goal = info_dict["goal"]
    planner_info["goal"] = goal[:, -1:] if goal.ndim >= 5 else goal.unsqueeze(1)

    past_blocks = []
    for offset in range(history_len - 1):
        start = offset * action_block + (action_block - 1)
        end = start + action_block
        past_blocks.append(
            raw_actions[:, start:end].reshape(raw_actions.size(0), 1, -1)
        )

    if past_blocks:
        planner_info["action"] = torch.cat(past_blocks, dim=1)
    else:
        planner_info["action"] = raw_actions.new_zeros(
            raw_actions.size(0), 0, block_action_dim
        )

    return planner_info


class BoundedCEMSolver(swm.solver.CEMSolver):
    """CEM solver that keeps normalized action candidates inside env bounds."""

    def set_normalized_action_bounds(self, low: np.ndarray, high: np.ndarray) -> None:
        low_t = torch.as_tensor(low, device=self.device, dtype=torch.float32)
        high_t = torch.as_tensor(high, device=self.device, dtype=torch.float32)
        self._normalized_low = low_t.reshape(self.n_envs, 1, 1, self.action_dim)
        self._normalized_high = high_t.reshape(self.n_envs, 1, 1, self.action_dim)

    @classmethod
    def from_solver(cls, solver):
        bounded = cls(
            model=solver.model,
            batch_size=solver.batch_size,
            num_samples=solver.num_samples,
            var_scale=solver.var_scale,
            n_steps=solver.n_steps,
            topk=solver.topk,
            device=solver.device,
        )
        bounded.torch_gen = solver.torch_gen
        return bounded

    def _clip_candidates(self, candidates: torch.Tensor, start_idx: int, end_idx: int) -> torch.Tensor:
        low = getattr(self, "_normalized_low", None)
        high = getattr(self, "_normalized_high", None)
        if low is None or high is None:
            return candidates
        return torch.clamp(candidates, low[start_idx:end_idx], high[start_idx:end_idx])

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        outputs = {"costs": [], "mean": [], "var": []}

        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        total_envs = self.n_envs
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx

            batch_mean = self._clip_candidates(mean[start_idx:end_idx].unsqueeze(1), start_idx, end_idx).squeeze(1)
            batch_var = var[start_idx:end_idx]

            batch_infos = {}
            for key, value in info_dict.items():
                value_batch = value[start_idx:end_idx]
                batch_infos[key] = value_batch

            final_batch_cost = None
            for _ in range(self.n_steps):
                candidates = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean
                candidates = self._clip_candidates(candidates, start_idx, end_idx)

                costs = self.model.get_cost(batch_infos.copy(), candidates)
                assert isinstance(costs, torch.Tensor), f"Expected cost to be a torch.Tensor, got {type(costs)}"
                assert costs.ndim == 2 and costs.shape == (current_bs, self.num_samples), (
                    f"Expected cost shape ({current_bs}, {self.num_samples}), got {costs.shape}"
                )

                topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[batch_indices, topk_inds]
                batch_mean = self._clip_candidates(topk_candidates.mean(dim=1).unsqueeze(1), start_idx, end_idx).squeeze(1)
                batch_var = topk_candidates.std(dim=1, unbiased=False).clamp_min(1e-6)
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = self._clip_candidates(mean.unsqueeze(1), 0, total_envs).squeeze(1).detach().cpu()
        outputs["mean"] = [outputs["actions"]]
        outputs["var"] = [var.detach().cpu()]
        print(f"CEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs


def find_stacked_wrapper(env):
    current = env
    for _ in range(10):
        if current is None:
            return None
        if current.__class__.__name__ == "StackedWrapper":
            return current
        current = getattr(current, "env", None)
    return None


def seed_history_buffers(
    world,
    history_payload: dict[str, np.ndarray],
    goal_history_payload: dict[str, np.ndarray],
) -> None:
    for env_idx, env in enumerate(world.envs.unwrapped.envs):
        stacked = find_stacked_wrapper(env)
        if stacked is None:
            continue

        for key, sequence in history_payload.items():
            if key not in stacked.buffers:
                continue
            buffer = stacked.buffers[key]
            buffer.clear()
            buffer.extend([sequence[env_idx, t] for t in range(sequence.shape[1])])

        for key, sequence in goal_history_payload.items():
            if key not in stacked.buffers:
                continue
            buffer = stacked.buffers[key]
            buffer.clear()
            buffer.extend([sequence[env_idx, t] for t in range(sequence.shape[1])])


def to_numpy_step(value: Any):
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.copy()
    return np.asarray(value)


def extract_dataset_eval_payload(
    data,
    columns,
    *,
    history_span: int,
    goal_offset_steps: int,
):
    init_step_per_env = {}
    goal_step_per_env = {}
    history_payload = {}
    goal_history_payload = {}

    for col in columns:
        history_values = []
        init_values = []
        goal_values = []

        for ep in data:
            series = ep[col]
            if col.startswith("pixels"):
                series = series.permute(0, 2, 3, 1)
            if isinstance(series, torch.Tensor):
                series = series.cpu().numpy()

            history = np.array(series[:history_span], copy=True)
            current = np.array(series[history_span - 1], copy=True)
            goal = np.array(series[history_span - 1 + goal_offset_steps], copy=True)

            history_values.append(history)
            init_values.append(current)
            goal_values.append(goal)

        history_payload[col] = np.stack(history_values)
        init_step_per_env[col] = np.stack(init_values)
        goal_key = "goal" if col == "pixels" else f"goal_{col}"
        goal_step_per_env[goal_key] = np.stack(goal_values)
        goal_history_payload[goal_key] = np.repeat(
            goal_step_per_env[goal_key][:, None, ...], history_span, axis=1
        )

    return init_step_per_env, goal_step_per_env, history_payload, goal_history_payload


def apply_dataset_callables(world, init_step, callables):
    callables = callables or []
    for env_idx, env in enumerate(world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        for spec in callables:
            method_name = spec["method"]
            required = bool(spec.get("required", True))
            if not hasattr(env_unwrapped, method_name):
                if required:
                    raise AttributeError(
                        f"Required eval callable '{method_name}' is missing on env {env_idx}"
                    )
                continue

            args = spec.get("args", spec)
            prepared_args = {}
            for arg_name, arg_data in args.items():
                value = arg_data.get("value", None)
                in_dataset = arg_data.get("in_dataset", True)
                if in_dataset:
                    if value not in init_step:
                        if required:
                            raise KeyError(
                                f"Required eval callable '{method_name}' needs dataset key "
                                f"'{value}' for argument '{arg_name}'"
                            )
                        continue
                    prepared_args[arg_name] = to_numpy_step(init_step[value][env_idx])
                else:
                    prepared_args[arg_name] = value
            getattr(env_unwrapped, method_name)(**prepared_args)


def sync_dataset_goal_rendering(world, goal_state, goal_image=None):
    """Keep PushT's rendered target overlay aligned with dataset goal_state."""

    if goal_state is None:
        return

    for env_idx, env in enumerate(world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        current_goal_state = np.asarray(to_numpy_step(goal_state[env_idx])).reshape(-1)
        if hasattr(env_unwrapped, "_set_goal_state"):
            env_unwrapped._set_goal_state(current_goal_state.copy())
        elif hasattr(env_unwrapped, "goal_state"):
            env_unwrapped.goal_state = current_goal_state.copy()
        if current_goal_state.size >= 5 and hasattr(env_unwrapped, "goal_pose"):
            env_unwrapped.goal_pose = current_goal_state[2:5].copy()
        if goal_image is not None and hasattr(env_unwrapped, "_goal"):
            env_unwrapped._goal = to_numpy_step(goal_image[env_idx])


def evaluate_from_dataset_fixed(
    world,
    dataset,
    *,
    episodes_idx,
    start_steps,
    goal_offset_steps,
    eval_budget,
    callables,
    save_video,
    video_path,
    history_span,
):
    ep_idx_arr = np.asarray(episodes_idx)
    start_steps_arr = np.asarray(start_steps)
    history_start = start_steps_arr - (history_span - 1)
    end_steps = start_steps_arr + goal_offset_steps + 1

    data = dataset.load_chunk(ep_idx_arr, history_start, end_steps)
    columns = dataset.column_names
    init_step, goal_step, history_payload, goal_history_payload = (
        extract_dataset_eval_payload(
            data,
            columns,
            history_span=history_span,
            goal_offset_steps=goal_offset_steps,
        )
    )

    seeds = init_step.get("seed")
    if seeds is not None:
        seeds = [int(v) for v in np.asarray(seeds).tolist()]

    world.reset(seed=seeds)
    callable_state = deepcopy(init_step)
    callable_state.update(deepcopy(goal_step))
    apply_dataset_callables(world, callable_state, callables)
    sync_dataset_goal_rendering(
        world,
        goal_step.get("goal_state"),
        goal_step.get("goal"),
    )

    world.infos.update({k: v.copy() for k, v in history_payload.items()})
    world.infos.update({k: v.copy() for k, v in goal_history_payload.items()})
    seed_history_buffers(world, history_payload, goal_history_payload)

    results = {
        "success_rate": 0.0,
        "episode_successes": np.zeros(len(ep_idx_arr), dtype=bool),
        "seeds": np.asarray(seeds) if seeds is not None else None,
    }

    video_frames = None
    if save_video:
        video_frames = np.empty(
            (world.num_envs, eval_budget, *world.infos["pixels"].shape[-3:]),
            dtype=np.uint8,
        )

    for step_idx in range(eval_budget):
        if save_video:
            video_frames[:, step_idx] = world.infos["pixels"][:, -1]
        world.infos.update({k: v.copy() for k, v in goal_history_payload.items()})
        world.step()
        results["episode_successes"] = np.logical_or(
            results["episode_successes"], np.asarray(world.terminateds, dtype=bool)
        )
        world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))

    if save_video:
        video_frames[:, -1] = world.infos["pixels"][:, -1]
    results["success_rate"] = (
        float(np.sum(results["episode_successes"])) / len(ep_idx_arr) * 100.0
    )

    if save_video:
        import imageio

        video_path = Path(video_path)
        video_path.mkdir(parents=True, exist_ok=True)
        goal_frames = goal_history_payload["goal"][:, -1]
        for env_idx in range(world.num_envs):
            out = imageio.get_writer(
                video_path / f"rollout_{env_idx}.mp4",
                fps=15,
                codec="libx264",
            )
            goals = np.vstack([goal_frames[env_idx], goal_frames[env_idx]])
            for t in range(eval_budget):
                stacked_frame = np.vstack(
                    [video_frames[env_idx, t], goal_frames[env_idx]]
                )
                out.append_data(np.hstack([stacked_frame, goals]))
            out.close()

    return results


class LeWMEvalPolicy(swm.policy.WorldModelPolicy):
    def __init__(self, *args, model_history_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_history_size = model_history_size

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        if not hasattr(self.solver, "set_normalized_action_bounds"):
            return

        if hasattr(env, "single_action_space"):
            raw_low = np.asarray(env.single_action_space.low, dtype=np.float32).reshape(-1)
            raw_high = np.asarray(env.single_action_space.high, dtype=np.float32).reshape(-1)
        else:
            raw_low = np.asarray(env.action_space.low, dtype=np.float32)
            raw_high = np.asarray(env.action_space.high, dtype=np.float32)
            if raw_low.ndim > 1:
                raw_low = raw_low[0]
                raw_high = raw_high[0]
            raw_low = raw_low.reshape(-1)
            raw_high = raw_high.reshape(-1)

        if "action" in self.process:
            normalized_bounds = self.process["action"].transform(np.stack([raw_low, raw_high]))
            low = np.minimum(normalized_bounds[0], normalized_bounds[1])
            high = np.maximum(normalized_bounds[0], normalized_bounds[1])
        else:
            low = np.minimum(raw_low, raw_high)
            high = np.maximum(raw_low, raw_high)

        block_low = np.repeat(np.tile(low, self.cfg.action_block)[None, :], env.num_envs, axis=0)
        block_high = np.repeat(np.tile(high, self.cfg.action_block)[None, :], env.num_envs, axis=0)
        self.solver.set_normalized_action_bounds(block_low, block_high)

    def get_action(self, info_dict: dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        if len(self._action_buffer) == 0:
            prepared = self._prepare_info(info_dict)
            planner_info = build_planning_context(
                prepared,
                history_len=self.model_history_size,
                action_block=self.cfg.action_block,
                block_action_dim=self.solver.action_dim,
            )
            outputs = self.solver(planner_info, init_action=self._next_init)

            actions = outputs["actions"]
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]
            self._next_init = rest if self.cfg.warm_start else None
            plan = plan.reshape(self.env.num_envs, self.flatten_receding_horizon, -1)
            self._action_buffer.extend(plan.transpose(0, 1))

        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape)
        action = action.numpy()

        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)

        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of world-model planning on PushT."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"
    assert (
        cfg.eval.goal_offset_steps == cfg.plan_config.horizon * cfg.plan_config.action_block
    ), "Goal offset must match the planned raw-step horizon"

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    policy = cfg.get("policy", "random")
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    normalizer_episodes, normalizer_metadata = resolve_normalizer_scope(dataset, policy)
    process = build_processors(cfg, dataset, normalizer_episodes)
    model_history_size = 1
    history_span = 1
    actual_solver_metadata = {}

    if policy != "random":
        device = cfg.solver.device
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        model_history_size = infer_model_history_size(model)
        history_span = model_history_size * cfg.plan_config.action_block
        cfg.world.history_size = history_span
        cfg.world.frame_skip = 1

        plan_config = OmegaConf.to_container(cfg.plan_config, resolve=True)
        plan_config["history_len"] = model_history_size
        config = swm.PlanConfig(**plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        if isinstance(solver, swm.solver.CEMSolver) and not isinstance(solver, BoundedCEMSolver):
            solver = BoundedCEMSolver.from_solver(solver)
        actual_solver_metadata = {
            "solver_class": f"{solver.__class__.__module__}.{solver.__class__.__name__}",
            "solver_batch_size": int(getattr(solver, "batch_size", -1)),
            "solver_num_samples": int(getattr(solver, "num_samples", -1)),
            "solver_n_steps": int(getattr(solver, "n_steps", -1)),
            "solver_topk": int(getattr(solver, "topk", -1)),
        }
        policy = LeWMEvalPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
            model_history_size=model_history_size,
        )
    else:
        policy = swm.policy.RandomPolicy()

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    results_dir = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    step_idx = dataset.get_col_data("step_idx")
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    min_start_idx = history_span - 1
    valid_mask = (step_idx >= min_start_idx) & (step_idx <= max_start_per_row)
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")
    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError(
            f"Not enough valid starting points for evaluation: "
            f"requested {cfg.eval.num_eval}, found {len(valid_indices)}"
        )

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices), size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_rows = dataset.get_row_data(random_episode_indices)
    eval_episodes = eval_rows[col_name]
    eval_start_idx = eval_rows["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    with open_dict(cfg):
        cfg.eval.normalizer_metadata = normalizer_metadata
        if actual_solver_metadata:
            cfg.eval.actual_solver = actual_solver_metadata

    start_time = time.time()
    metrics = evaluate_from_dataset_fixed(
        world,
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        save_video=cfg.output.get("save_video", False),
        video_path=results_dir,
        history_span=history_span,
    )
    metrics.update(
        {
            "eval_row_indices": random_episode_indices.tolist(),
            "eval_episodes": np.asarray(eval_episodes).tolist(),
            "eval_start_idx": np.asarray(eval_start_idx).tolist(),
            "normalizer_metadata": normalizer_metadata,
            **actual_solver_metadata,
        }
    )
    end_time = time.time()

    print(metrics)

    results_path = results_dir / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w") as f:
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"metrics_json: {json.dumps(to_jsonable(metrics), sort_keys=True)}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
