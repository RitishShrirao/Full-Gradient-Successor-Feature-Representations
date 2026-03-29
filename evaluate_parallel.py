import os
import argparse
import logging
import imageio
import numpy as np
import random
import json
import re
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Safer defaults for headless MuJoCo rendering.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# Plotting is optional.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from utils.utils import load_agent_weights
from agents.buffer import ReplayBuffer, ConditionalReplayBuffer
from agents.dqn import DQN
from agents.sfdqn import SFDQN
from agents.fgsfdqn import FGSFDQN
from agents.fgdqn import FGDQN
from train_parallel import MLP, AvgFGSFDQN, load_config, build_task_sequence, setup_configs
from features.deep import DeepSF
from features.deep_fg import DeepFGSF


def _load_tasks_meta(weight_root):
    meta_path = os.path.join(weight_root, 'tasks', 'tasks_meta.json')
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read tasks metadata at {meta_path}: {e}")
        return None


def _apply_tasks_meta_to_cfg(cfg, meta):
    if not meta:
        return cfg

    env = str(meta.get('env', cfg['TASK'].get('env', 'gridworld'))).strip().lower()
    cfg['TASK']['env'] = env
    if 'n_tasks' in meta:
        cfg['GENERAL']['n_tasks'] = str(int(meta['n_tasks']))

    if env == 'maze':
        cfg['TASK']['env'] = 'maze'
        if 'env_id' in meta:
            cfg['TASK']['maze_env_id'] = str(meta['env_id'])
        if 'effective_goal_cells_per_task' in meta:
            cfg['TASK']['maze_goal_cells'] = repr([tuple(c) for c in meta['effective_goal_cells_per_task']])
        elif 'goal_cells' in meta:
            cfg['TASK']['maze_goal_cells'] = repr([tuple(c) for c in meta['goal_cells']])
        if 'reset_cell' in meta and meta['reset_cell'] is not None:
            cfg['TASK']['maze_reset_cell'] = repr(tuple(meta['reset_cell']))
        if 'include_goal_in_state' in meta:
            cfg['TASK']['maze_include_goal_in_state'] = str(bool(meta['include_goal_in_state']))
        if 'goal_jitter' in meta:
            cfg['TASK']['maze_goal_jitter'] = str(float(meta['goal_jitter']))

    return cfg


def _extract_trial_index(path):
    m = re.search(r"trial_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def _list_weight_files(weight_dir):
    files = [
        os.path.join(weight_dir, f)
        for f in os.listdir(weight_dir)
        if f.endswith(".pt")
    ]
    return sorted(files, key=lambda p: (_extract_trial_index(p), os.path.basename(p)))


def _checkpoint_policy_count(weight_file):
    """Return number of SF policies in a checkpoint, or None."""
    try:
        ckpt = torch.load(weight_file, map_location="cpu", weights_only=False)
    except Exception:
        return None

    psi = ckpt.get("psi_networks")
    if isinstance(psi, dict):
        return len(psi)
    return None


def _is_sf_model(model_name):
    return model_name in {
        "SFDQN", "SFDQN", "SFDQN_SFR",
        "FGSFDQN", "FGSFDQN", "FGSFDQN_SFR",
        "alg2", "alg3", "AvgFGSFDQN",
    }


def _filter_compatible_checkpoints(weight_files, model_name, expected_tasks):
    """Keep checkpoints matching the current SF task count."""
    if not _is_sf_model(model_name):
        return list(weight_files)

    compatible = []
    for wf in weight_files:
        n_policies = _checkpoint_policy_count(wf)
        if n_policies is None:
            logger.warning(
                f"Could not inspect checkpoint {os.path.basename(wf)} for {model_name}; skipping for safety"
            )
            continue
        if n_policies != expected_tasks:
            logger.warning(
                f"Skipping incompatible checkpoint {os.path.basename(wf)} for {model_name}: "
                f"checkpoint policies={n_policies}, current tasks={expected_tasks}"
            )
            continue
        compatible.append(wf)
    return compatible


def _set_eval_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_single_checkpoint_run(
    model_name,
    config_path,
    weight_file,
    n_episodes,
    max_steps,
    seed,
):
    """Evaluate one checkpoint for one eval run."""
    _set_eval_seed(seed)
    cfg = load_config(config_path)
    inferred_root = os.path.dirname(os.path.dirname(weight_file))
    cfg = _apply_tasks_meta_to_cfg(cfg, _load_tasks_meta(inferred_root))
    tasks = build_task_sequence(cfg)
    agent = build_agent_by_name(model_name, cfg, tasks)
    try:
        load_agent_weights(agent, weight_file)
    except AssertionError as e:
        raise RuntimeError(f"Checkpoint/task mismatch for {os.path.basename(weight_file)}: {e}")

    n_tasks = len(tasks)
    rewards_arr = np.zeros((n_tasks, n_episodes), dtype=np.float32)
    for task_idx, task in enumerate(tasks):
        if hasattr(agent, 'set_active_training_task'):
            try:
                agent.set_active_training_task(task_idx, reset=False)
            except Exception:
                pass
        episodes = evaluate_agent(
            agent,
            task,
            n_episodes=n_episodes,
            max_steps=max_steps,
            collect_frames=False,
        )
        rewards = [ep['reward'] for ep in episodes]
        if len(rewards) < n_episodes:
            rewards = rewards + [0.0] * (n_episodes - len(rewards))
        rewards_arr[task_idx, :] = np.asarray(rewards[:n_episodes], dtype=np.float32)

    return rewards_arr


def render_frame_from_task(task):
    env = getattr(task, 'env', None)
    if env is None:
        return None

    # Force rgb_array render mode.
    try:
        if hasattr(env, "render_mode"):
            env.render_mode = "rgb_array"
    except Exception as e:
        logging.debug(f"could not set render_mode attribute: {e}")

    try:
        frame = env.render()
        if isinstance(frame, tuple) and len(frame) > 0:
            frame = frame[0]
        return frame
    except Exception as e:
        logging.debug(f"render_frame env.render() failed: {e}")

    # Fallback to unwrapped env.
    try:
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None:
            frame = unwrapped.render()
            if isinstance(frame, tuple) and len(frame) > 0:
                frame = frame[0]
            return frame
    except Exception as e:
        logging.debug(f"render_frame unwrapped.render() failed: {e}")

    return None


def _normalize_frame(frame):
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr] * 3, axis=2)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


def evaluate_agent(agent, task, n_episodes=3, max_steps=None, collect_frames=False):
    """Run eval episodes and optionally collect frames."""
    episodes = []
    # Try to configure rendering.
    env = getattr(task, 'env', None)
    if env is not None:
        try:
            if hasattr(env, "render_mode"):
                env.render_mode = "rgb_array"
        except Exception as e:
            logging.debug(f"evaluate_agent: could not set env.render_mode: {e}")
        try:
            env.reset(render_mode='rgb_array')
        except Exception as e:
            logging.debug(f"evaluate_agent: additional reset(render_mode) failed: {e}")
        try:
            env.reset(options={'render_mode':'rgb_array'})
        except Exception as e:
            logging.debug(f"evaluate_agent: additional reset(options) failed: {e}")

    for ep in range(n_episodes):
        s = task.initialize()
        s_enc = agent.encoding(s) if hasattr(agent, 'encoding') else s
        frames = []
        if collect_frames:
            f = render_frame_from_task(task)
            if f is not None:
                frames.append(_normalize_frame(f))

        T = max_steps if max_steps is not None else getattr(agent, 'T', 200)
        total_reward = 0.0
        for t in range(T):
            if hasattr(agent, 'get_test_action'):
                try:
                    a = agent.get_test_action(s_enc)
                except TypeError:
                    # Some agents require reward weights.
                    try:
                        w = task.get_w()
                        a = agent.get_test_action(s_enc, w)
                    except Exception:
                        a = np.random.randint(0, task.action_count()) if hasattr(task, 'action_count') else 0
            else:
                a = np.random.randint(0, task.action_count()) if hasattr(task, 'action_count') else 0

            s1, r, done = task.transition(a)
            total_reward += r
            s_enc = agent.encoding(s1) if hasattr(agent, 'encoding') else s1
            if collect_frames:
                f = render_frame_from_task(task)
                if f is not None:
                    frames.append(_normalize_frame(f))
            if done:
                break
            s = s1

        episodes.append({'reward': total_reward, 'frames': frames})
    return episodes


def build_agent_by_name(name, cfg, tasks):
    input_dim = tasks[0].encode_dim()
    n_actions = tasks[0].action_count()
    n_features = tasks[0].feature_dim()

    reward_model = cfg['SFQL'].get('reward_model', 'linear')
    successor_representation = cfg['SFQL'].get('successor_representation', 'sf')
    reward_hidden_units = int(cfg['SFQL'].get('reward_hidden_units', 128))
    reward_learning_rate = float(cfg['SFQL'].get('reward_learning_rate', 1e-3))
    sfr_max_centers = int(cfg['SFQL'].get('sfr_max_centers', 64))
    sfr_center_threshold = float(cfg['SFQL'].get('sfr_center_threshold', 0.0))

    # Backward-compatible aliases.
    if name in {'SFDQN', 'FGSFDQN'}:
        reward_model = 'linear'
        successor_representation = 'sf'
    elif name in {'SFDQN_SFR', 'FGSFDQN_SFR'}:
        reward_model = 'nonlinear'
        successor_representation = 'sfr'

    sf_dim = sfr_max_centers if successor_representation == 'sfr' else n_features

    if name == 'DQN':
        def model_builder():
            return MLP(input_dim, n_actions, learning_rate=float(cfg['QL'].get('learning_rate', 0.5)))
        buffer = ReplayBuffer(n_samples=int(cfg['GENERAL'].get('buffer_size', 200000)), n_batch=int(cfg['GENERAL'].get('n_batch', 512)))
        agent = DQN(model_builder, buffer, gamma=float(cfg['AGENT'].get('gamma', 0.95)), epsilon=float(cfg['AGENT'].get('epsilon', 0.55)), T=int(cfg['AGENT'].get('T', 200)), encoding=tasks[0].encode, save_ev=int(cfg['AGENT'].get('save_ev', 200)))

    elif name in {'SFDQN', 'SFDQN', 'SFDQN_SFR'}:
        sf = DeepSF(
            input_dim=input_dim,
            n_actions=n_actions,
            n_features=sf_dim,
            learning_rate=float(cfg['SFQL'].get('learning_rate', 0.001)),
            learning_rate_w=float(cfg['SFQL'].get('learning_rate_w', 0.5)),
            reward_model=reward_model,
            reward_hidden_units=reward_hidden_units,
            reward_learning_rate=reward_learning_rate,
            successor_representation=successor_representation,
            reward_input_dim=n_features,
            sfr_center_threshold=sfr_center_threshold,
        )
        buffer = ReplayBuffer(n_samples=int(cfg['GENERAL'].get('buffer_size', 200000)), n_batch=int(cfg['GENERAL'].get('n_batch', 512)))
        agent = SFDQN(sf, buffer, gamma=float(cfg['AGENT'].get('gamma', 0.95)), epsilon=float(cfg['AGENT'].get('epsilon', 0.55)), T=int(cfg['AGENT'].get('T', 200)), encoding=tasks[0].encode, save_ev=int(cfg['AGENT'].get('save_ev', 200)))

    elif name in {'FGSFDQN', 'FGSFDQN', 'FGSFDQN_SFR', 'alg2', 'alg3'}:
        sf = DeepFGSF(
            input_dim=input_dim,
            n_actions=n_actions,
            n_features=sf_dim,
            learning_rate=float(cfg['SFQL'].get('learning_rate', 0.001)),
            learning_rate_prior=float(cfg['SFQL'].get('learning_rate_prior', 1e-5)),
            learning_rate_w=float(cfg['SFQL'].get('learning_rate_w', 0.5)),
            reward_model=reward_model,
            reward_hidden_units=reward_hidden_units,
            reward_learning_rate=reward_learning_rate,
            successor_representation=successor_representation,
            reward_input_dim=n_features,
            sfr_center_threshold=sfr_center_threshold,
        )
        if name == 'alg3':
            buffer = ConditionalReplayBuffer(n_samples=int(cfg['GENERAL'].get('buffer_size', 200000)), n_batch=int(cfg['GENERAL'].get('n_batch', 512)))
        else:
            buffer = ReplayBuffer(n_samples=int(cfg['GENERAL'].get('buffer_size', 200000)), n_batch=int(cfg['GENERAL'].get('n_batch', 512)))
        algo = 'alg1' if name == 'FGSFDQN' else name
        if name in {'FGSFDQN', 'FGSFDQN_SFR'}:
            algo = 'alg1'
        n_avg = 1 if name in ['FGSFDQN', 'FGSFDQN_SFR', 'alg2'] else 5
        agent = FGSFDQN(sf, buffer, gamma=float(cfg['AGENT'].get('gamma', 0.95)), epsilon=float(cfg['AGENT'].get('epsilon', 0.55)), T=int(cfg['AGENT'].get('T', 200)), encoding='task', algorithm=algo, n_averaging=n_avg, save_ev=int(cfg['AGENT'].get('save_ev', 200)))

    elif name == 'AvgFGSFDQN':
        sf = DeepFGSF(
            input_dim=input_dim,
            n_actions=n_actions,
            n_features=sf_dim,
            learning_rate=float(cfg['SFQL'].get('learning_rate', 0.001)),
            learning_rate_prior=float(cfg['SFQL'].get('learning_rate_prior', 1e-5)),
            learning_rate_w=float(cfg['SFQL'].get('learning_rate_w', 0.5)),
            reward_model=reward_model,
            reward_hidden_units=reward_hidden_units,
            reward_learning_rate=reward_learning_rate,
            successor_representation=successor_representation,
            reward_input_dim=n_features,
            sfr_center_threshold=sfr_center_threshold,
        )
        buffer = ConditionalReplayBuffer(n_samples=int(cfg['GENERAL'].get('buffer_size', 200000)), n_batch=int(cfg['GENERAL'].get('n_batch', 512)))
        agent = AvgFGSFDQN(sf, buffer, gamma=float(cfg['AGENT'].get('gamma', 0.95)), T=int(cfg['AGENT'].get('T', 200)), epsilon=float(cfg['AGENT'].get('epsilon', 0.55)), encoding='task', algorithm='alg1_averaged', save_ev=int(cfg['AGENT'].get('save_ev', 200)))

    elif name == 'FGDQN':
        def model_builder():
            return MLP(input_dim, n_actions, learning_rate=float(cfg['QL'].get('learning_rate', 0.5)))
        buffer = ReplayBuffer(n_samples=int(cfg['GENERAL'].get('buffer_size', 200000)), n_batch=int(cfg['GENERAL'].get('n_batch', 512)))
        agent = FGDQN(model_builder, buffer, gamma=float(cfg['AGENT'].get('gamma', 0.95)), epsilon=float(cfg['AGENT'].get('epsilon', 0.55)), T=int(cfg['AGENT'].get('T', 200)), encoding=tasks[0].encode, save_ev=int(cfg['AGENT'].get('save_ev', 200)))
    else:
        raise ValueError(f'Unknown agent: {name}')

    agent.reset()
    for t in tasks:
        try:
            agent.add_training_task(t)
        except Exception:
            pass

    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.cfg')
    parser.add_argument('--weight-root', default='updated_weights')
    parser.add_argument('--out-root', default='videos')
    parser.add_argument('--models', nargs='+', default=[
        'DQN',
        'SFDQN', 'SFDQN_SFR',
        'FGSFDQN', 'FGSFDQN_SFR',
        'alg2', 'alg3',
        'FGDQN'
    ])
    parser.add_argument('--n-episodes', type=int, default=2)
    parser.add_argument('--n-eval-runs', type=int, default=3,
                        help='Number of repeated evaluation passes per checkpoint.')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Optional cap on steps per episode. Defaults to agent.T if omitted.')
    parser.add_argument('--seed', type=int, default=123,
                        help='Base seed used for repeatable evaluations.')
    parser.add_argument('--n-workers', type=int, default=16,
                        help='Number of worker processes for parallel evaluation jobs.')
    parser.add_argument('--checkpoint-mode', choices=['all', 'latest', 'first'], default='all',
                        help='Which checkpoint(s) to evaluate for each model.')
    parser.add_argument('--video-checkpoint', choices=['none', 'first', 'latest'], default='latest',
                        help='Select which checkpoint to render videos for (run 0 only).')
    args = parser.parse_args()

    if not os.path.exists('configs'):
        setup_configs()

    cfg = load_config(args.config)
    cfg = _apply_tasks_meta_to_cfg(cfg, _load_tasks_meta(args.weight_root))

    os.makedirs(args.out_root, exist_ok=True)

    results = {}
    summary = {}
    for model_name in args.models:
        weight_dir = os.path.join(args.weight_root, model_name)
        if not os.path.isdir(weight_dir):
            logger.warning(f"No weight files found for {model_name} in {weight_dir}; skipping evaluation")
            continue

        weight_files = _list_weight_files(weight_dir)
        if not weight_files:
            logger.warning(f"No weight files found for {model_name} in {weight_dir}; skipping evaluation")
            continue

        logger.info(f"Evaluating {model_name}")
        tasks = build_task_sequence(cfg)
        n_tasks = len(tasks)

        compatible_weights = _filter_compatible_checkpoints(weight_files, model_name, n_tasks)
        if not compatible_weights:
            logger.warning(
                f"No compatible checkpoints found for {model_name} with n_tasks={n_tasks}; skipping."
            )
            continue

        if args.checkpoint_mode == 'first':
            selected_weights = [compatible_weights[0]]
        elif args.checkpoint_mode == 'latest':
            selected_weights = [compatible_weights[-1]]
        else:
            selected_weights = compatible_weights

        logger.info(
            f"Evaluating {model_name} on {len(selected_weights)} checkpoint(s), "
            f"{args.n_eval_runs} run(s) each, {args.n_episodes} episode(s) per task"
        )

        # Try to rebuild UMAZE envs with rgb_array rendering.
        try:
            import gymnasium as gym
            import gymnasium_robotics
            gym.register_envs(gymnasium_robotics)
            for t in tasks:
                if hasattr(t, 'env_id'):
                    try:
                        t.env = gym.make(t.env_id, render_mode='rgb_array')
                    except Exception:
                        if hasattr(t.env, 'render_mode'):
                            t.env.render_mode = 'rgb_array'
        except ImportError:
            pass
        n_ckpt = len(selected_weights)
        reward_tensor = np.zeros((n_ckpt, args.n_eval_runs, n_tasks, args.n_episodes), dtype=np.float32)

        # Choose checkpoint for representative videos.
        video_ckpt_idx = None
        if args.video_checkpoint == 'first':
            video_ckpt_idx = 0
        elif args.video_checkpoint == 'latest':
            video_ckpt_idx = n_ckpt - 1

        total_jobs = n_ckpt * args.n_eval_runs
        n_workers = max(1, min(args.n_workers, total_jobs))
        logger.info(f"  Running {total_jobs} eval jobs with {n_workers} worker(s)")

        if n_workers == 1:
            for ckpt_idx, weight_file in enumerate(selected_weights):
                for run_idx in range(args.n_eval_runs):
                    run_seed = args.seed + (ckpt_idx * 1000) + run_idx
                    rewards_arr = _evaluate_single_checkpoint_run(
                        model_name=model_name,
                        config_path=args.config,
                        weight_file=weight_file,
                        n_episodes=args.n_episodes,
                        max_steps=args.max_steps,
                        seed=run_seed,
                    )
                    reward_tensor[ckpt_idx, run_idx, :, :] = rewards_arr
                    logger.info(
                        f"    done ckpt={os.path.basename(weight_file)} run={run_idx+1}/{args.n_eval_runs}"
                    )
        else:
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                future_map = {}
                for ckpt_idx, weight_file in enumerate(selected_weights):
                    for run_idx in range(args.n_eval_runs):
                        run_seed = args.seed + (ckpt_idx * 1000) + run_idx
                        fut = executor.submit(
                            _evaluate_single_checkpoint_run,
                            model_name,
                            args.config,
                            weight_file,
                            args.n_episodes,
                            args.max_steps,
                            run_seed,
                        )
                        future_map[fut] = (ckpt_idx, run_idx, weight_file)

                for fut in as_completed(future_map):
                    ckpt_idx, run_idx, weight_file = future_map[fut]
                    try:
                        rewards_arr = fut.result()
                        reward_tensor[ckpt_idx, run_idx, :, :] = rewards_arr
                        logger.info(
                            f"    done ckpt={os.path.basename(weight_file)} run={run_idx+1}/{args.n_eval_runs}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Evaluation failed for {model_name} ckpt={os.path.basename(weight_file)} "
                            f"run={run_idx+1}: {e}"
                        )

        # Optional representative videos.
        if video_ckpt_idx is not None:
            video_ckpt_path = selected_weights[video_ckpt_idx]
            logger.info(f"  Generating representative videos from {os.path.basename(video_ckpt_path)}")
            video_agent = build_agent_by_name(model_name, cfg, tasks)
            load_agent_weights(video_agent, video_ckpt_path)
            _set_eval_seed(args.seed)
            for task_idx, task in enumerate(tasks):
                if hasattr(video_agent, 'set_active_training_task'):
                    try:
                        video_agent.set_active_training_task(task_idx, reset=False)
                    except Exception:
                        pass
                episodes = evaluate_agent(
                    video_agent,
                    task,
                    n_episodes=args.n_episodes,
                    max_steps=args.max_steps,
                    collect_frames=True,
                )

                task_dir = os.path.join(args.out_root, model_name)
                os.makedirs(task_dir, exist_ok=True)
                tag = os.path.splitext(os.path.basename(video_ckpt_path))[0]
                out_path = os.path.join(task_dir, f'{tag}_task_{task_idx}.mp4')

                all_frames = []
                for ep in episodes:
                    all_frames.extend(ep['frames'])

                if not all_frames:
                    logger.warning(f"No frames collected for {model_name} task {task_idx}; skipping video")
                    continue

                try:
                    from moviepy.editor import ImageSequenceClip
                    clip = ImageSequenceClip([np.asarray(f) for f in all_frames], fps=args.fps)
                    clip.write_videofile(out_path, codec='libx264', audio=False, verbose=False, logger=None)
                except Exception:
                    try:
                        imageio.mimwrite(out_path, all_frames, fps=args.fps)
                    except Exception as e:
                        # Last fallback: save PNG frames.
                        seq_dir = out_path + '_frames'
                        os.makedirs(seq_dir, exist_ok=True)
                        for i, fr in enumerate(all_frames):
                            imageio.imwrite(os.path.join(seq_dir, f'frame_{i:04d}.png'), fr)
                        # Try ffmpeg if available.
                        try:
                            from shutil import which
                            if which('ffmpeg'):
                                import subprocess
                                cmd = [
                                    'ffmpeg', '-y', '-framerate', str(args.fps), '-i', os.path.join(seq_dir, 'frame_%04d.png'),
                                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_path
                                ]
                                subprocess.run(cmd, check=True)
                                logger.info(f"Encoded frames to {out_path}")
                            else:
                                logger.warning(f"ffmpeg not found; frames saved to {seq_dir}")
                        except Exception as e2:
                            logger.warning(f"Could not write video; frames saved to {seq_dir} ({e2})")
                logger.info(f"Saved {out_path}")

        # Flatten checkpoints/runs/episodes into samples.
        flat_samples = np.transpose(reward_tensor, (0, 1, 3, 2)).reshape(-1, n_tasks)
        results[model_name] = flat_samples

        per_trial_stats = []
        for ckpt_idx, weight_file in enumerate(selected_weights):
            trial_samples = np.transpose(reward_tensor[ckpt_idx], (0, 2, 1)).reshape(-1, n_tasks)
            per_trial_stats.append({
                'checkpoint': os.path.basename(weight_file),
                'task_mean': np.mean(trial_samples, axis=0).tolist() if trial_samples.size else [],
                'task_variance': np.var(trial_samples, axis=0).tolist() if trial_samples.size else [],
                'overall_mean': float(np.mean(trial_samples)) if trial_samples.size else float('nan'),
                'overall_variance': float(np.var(trial_samples)) if trial_samples.size else float('nan'),
            })

        summary[model_name] = {
            'n_checkpoints': int(n_ckpt),
            'n_eval_runs': int(args.n_eval_runs),
            'n_episodes': int(args.n_episodes),
            'task_mean': np.mean(flat_samples, axis=0).tolist() if flat_samples.size else [],
            'task_variance': np.var(flat_samples, axis=0).tolist() if flat_samples.size else [],
            'overall_mean': float(np.mean(flat_samples)) if flat_samples.size else float('nan'),
            'overall_variance': float(np.var(flat_samples)) if flat_samples.size else float('nan'),
            'per_trial': per_trial_stats,
        }

    if results:
        np.savez_compressed(os.path.join(args.out_root, 'evaluation_raw.npz'), **results)
        with open(os.path.join(args.out_root, 'evaluation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved evaluation artifacts to {args.out_root}")

    if results and plt is not None and sns is not None:
        plot_lib = plt
        sns_lib = sns

        def plot_final_evaluation_bar(results, title="Final Evaluation (Mean ± Std)", filename=""):
            """Plot per-task mean/std bars for each model."""
            sns_lib.set_style("whitegrid")

            agent_names = list(results.keys())
            n_tasks = results[agent_names[0]].shape[1]

            means = np.array([results[name].mean(axis=0) for name in agent_names])
            stds  = np.array([results[name].std(axis=0)  for name in agent_names])

            x = np.arange(n_tasks)
            width = 0.8 / len(agent_names)

            plot_lib.figure(figsize=(14, 6))

            for i, name in enumerate(agent_names):
                plot_lib.bar(
                    x + i * width,
                    means[i],
                    width,
                    yerr=stds[i],
                    capsize=4,
                    label=name,
                    alpha=0.85
                )

            plot_lib.xlabel("Task", fontsize=13)
            plot_lib.ylabel("Average Reward", fontsize=13)
            plot_lib.title(title, fontsize=16)
            plot_lib.xticks(x + width * (len(agent_names) - 1) / 2,
                       [f"Task {i}" for i in range(n_tasks)])
            plot_lib.legend()
            plot_lib.tight_layout()
            if filename:
                plot_lib.savefig(filename, format='pdf', bbox_inches='tight')
            plot_lib.show()

        plot_final_evaluation_bar(results, filename=os.path.join(args.out_root, 'finaleval.pdf'))
    elif results:
        logger.warning("Skipping final bar plot because matplotlib/seaborn are not installed")


if __name__ == '__main__':
    main()
