import os
import ast
import configparser
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import traceback

try:
    import wandb
    _has_wandb = True
except ImportError:  # pragma: no cover
    wandb = None
    _has_wandb = False

from tasks.gridworld import Shapes
from tasks.umaze import Umaze
from features.deep import DeepSF
from features.deep_fg import DeepFGSF
from agents.buffer import ReplayBuffer, ConditionalReplayBuffer
from agents.dqn import DQN
from agents.fgdqn import FGDQN
from agents.sfdqn import SFDQN
from agents.fgsfdqn import FGSFDQN
from utils.utils import save_agent_weights
import pickle
import json

LOG_FILE = os.environ.get("FGSF_LOG_FILE", "logger.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler()
    ],
)

logger = logging.getLogger(__name__)


def _get_wandb_run_name(agent_name, trial_index, cfg=None):
    prefix = cfg["GENERAL"].get("wandb_group", "") if cfg is not None else ""
    name = f"{agent_name}_trial_{trial_index+1}"
    return f"{prefix}/{name}" if prefix else name


def _parse_config_value(value: str):
    """Convert config string into int/float/bool when possible."""
    if value is None:
        return None
    v = value.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except Exception:
            continue
    return v


def _cfg_to_dict(cfg):
    """Convert a ConfigParser object into a plain dict for wandb."""
    out = {}
    for section in cfg.sections():
        out[section] = {}
        for key, val in cfg[section].items():
            out[section][key] = _parse_config_value(val)
    return out


def _wandb_config_from_cfg(cfg):
    """Return the full config as a dict for wandb logging."""
    try:
        return _cfg_to_dict(cfg)
    except Exception:
        return {}


def set_global_seed(seed: int):
    """Set numpy/random/torch seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_wandb_run(agent_name, trial_index, cfg):
    """Start a wandb run for reward logging."""
    if not _has_wandb:
        return None

    try:
        run = wandb.init(
            project=cfg["GENERAL"].get("wandb_project", "FG-SFDQN"),
            group=cfg["GENERAL"].get("wandb_group", None),
            job_type=agent_name,
            name=_get_wandb_run_name(agent_name, trial_index, cfg),
            reinit=True,
            config=_wandb_config_from_cfg(cfg),
            save_code=False,
            settings=wandb.Settings(init_timeout=180),
        )
        return run
    except Exception as exc:
        logger.warning(f"wandb.init failed: {exc}")
        return None


def load_config(path="configs/config.cfg"):
    cfg = configparser.ConfigParser()
    if not cfg.read(path):
        raise FileNotFoundError(f"config file not found: {path}")
    return cfg

def _parse_tuple_list(raw_value):
    values = ast.literal_eval(raw_value)
    return [tuple(map(float, item)) for item in values]

def _build_gridworld_tasks(cfg):
    maze = np.array(ast.literal_eval(cfg["TASK"]["maze"]), dtype=str)

    rewards_pool = [
        {'1': 1.0, '2': 0.0, '3': 0.0},
        {'1': 0.0, '2': 1.0, '3': 0.0},
        {'1': 0.0, '2': 0.0, '3': 1.0},
        {'1': 1.0, '2': -1.0, '3': 0.0},
        {'1': 0.0, '2': 1.0, '3': -1.0}
    ]

    n_tasks = int(cfg["GENERAL"]["n_tasks"])
    slip_prob = cfg["TASK"].getfloat("slip_prob", fallback=0.0)
    tasks = []
    for i in range(n_tasks):
        rewards = rewards_pool[i % len(rewards_pool)]
        tasks.append(Shapes(maze=maze, shape_rewards=rewards, slip_prob=slip_prob))
    return tasks

def _build_reacher_tasks(cfg):
    try:
        from tasks.reacher import Reacher
    except Exception as e:
        raise ImportError(
            "Failed to import Reacher task. Ensure pybulletgym and its dependencies are installed."
        ) from e

    target_positions = _parse_tuple_list(
        cfg["TASK"].get("reacher_train_targets", "[(0.14, 0.0), (-0.14, 0.0), (0.0, 0.14), (0.0, -0.14)]")
    )
    include_target_in_state = cfg["TASK"].getboolean("reacher_include_target_in_state", fallback=False)
    slip_prob = cfg["TASK"].getfloat("slip_prob", fallback=0.0)
    allow_discrete_actions = True
    n_tasks = int(cfg["GENERAL"]["n_tasks"])

    tasks = []
    for i in range(n_tasks):
        task_index = i % len(target_positions)
        tasks.append(
            Reacher(
                target_positions=target_positions,
                task_index=task_index,
                include_target_in_state=include_target_in_state,
                allow_discrete_actions=allow_discrete_actions,
                slip_prob=slip_prob,
            )
        )
    return tasks

def _build_maze_tasks(cfg):
    raw_goals = ast.literal_eval(cfg["TASK"].get("maze_goal_cells", "[(1, 1), (1, 3), (3, 3)]"))
    goal_cells = [tuple(map(int, c)) for c in raw_goals]

    env_id = cfg["TASK"].get("maze_env_id", "PointMaze_Large-v3")
    reset_cell_raw = cfg["TASK"].get("maze_reset_cell", fallback=None)
    reset_cell = tuple(map(int, ast.literal_eval(reset_cell_raw))) if reset_cell_raw else None
    include_goal_in_state = cfg["TASK"].getboolean("maze_include_goal_in_state", fallback=True)
    allow_discrete_actions = True
    slip_prob = cfg["TASK"].getfloat("slip_prob", fallback=0.0)
    n_tasks = int(cfg["GENERAL"]["n_tasks"])

    # Drop goal cells that land in walls.
    def _quick_check(cell):
        try:
            tmp2 = Umaze(
                goal_cells=[cell],
                task_index=0,
                env_id=env_id,
                reset_cell=reset_cell,
                include_goal_in_state=include_goal_in_state,
                allow_discrete_actions=allow_discrete_actions,
            )
            un2 = tmp2.env
            while hasattr(un2, "env"):
                un2 = un2.env
            maze2 = getattr(un2, "maze", None)
            if maze2 is None:
                return True
            r, c = int(cell[0]), int(cell[1])
            return maze2.maze_map[r][c] != 1
        except Exception:
            return False
    filtered = []
    for g in goal_cells:
        if _quick_check(g):
            filtered.append(g)
        else:
            logger.warning(f"maze: user goal {g} lies in wall, ignoring")
    goal_cells = filtered

    # Check whether a candidate cell is free.
    def _cell_is_free(cell):
        try:
            tmp = Umaze(
                goal_cells=[cell],
                task_index=0,
                env_id=env_id,
                reset_cell=reset_cell,
                include_goal_in_state=include_goal_in_state,
                allow_discrete_actions=allow_discrete_actions,
            )
            un = tmp.env
            while hasattr(un, "env"):
                un = un.env
            maze = getattr(un, "maze", None)
            if maze is None:
                return True
            r, c = int(cell[0]), int(cell[1])
            return maze.maze_map[r][c] != 1
        except Exception:
            return False

    # Expand nearby goal cells if we need more tasks.
    similarity_radius = cfg["TASK"].getint("maze_similarity_radius", fallback=0)
    goal_jitter = cfg["TASK"].getfloat("maze_goal_jitter", fallback=0.0)

    if n_tasks > len(goal_cells) and similarity_radius > 0:
        expanded = list(goal_cells)
        idx = 0
        while len(expanded) < n_tasks and idx < len(expanded):
            base = expanded[idx]
            row, col = base
            for dr in range(-similarity_radius, similarity_radius + 1):
                for dc in range(-similarity_radius, similarity_radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    candidate = (row + dr, col + dc)
                    if candidate in expanded:
                        continue
                    if not _cell_is_free(candidate):
                        logger.debug(f"maze: skipping wall cell candidate {candidate}")
                        continue
                    expanded.append(candidate)
                    if len(expanded) >= n_tasks:
                        break
                if len(expanded) >= n_tasks:
                    break
            idx += 1
        goal_cells = expanded[:n_tasks]
        logger.info(
            f"maze: auto-expanded goal_cells to {goal_cells} using radius {similarity_radius}"
        )

    # If still short, repeat the last valid cell.
    if len(goal_cells) < n_tasks:
        logger.warning(
            f"maze: only {len(goal_cells)} valid goal cells found but {n_tasks} tasks requested; "
            "repeating last cell to meet requirement."
        )
        if goal_cells:
            goal_cells += [goal_cells[-1]] * (n_tasks - len(goal_cells))
        else:
            raise ValueError("maze: no valid goal cells available - check your maze layout")
    tasks = []
    for i in range(n_tasks):
        task_index = i % len(goal_cells)
        tasks.append(
            Umaze(
                goal_cells=goal_cells,
                task_index=task_index,
                env_id=env_id,
                reset_cell=reset_cell,
                include_goal_in_state=include_goal_in_state,
                allow_discrete_actions=allow_discrete_actions,
                goal_jitter=goal_jitter,
                slip_prob=slip_prob,
            )
        )
    return tasks

def build_task_sequence(cfg):
    """Build tasks for the selected environment."""
    env_name = cfg["TASK"].get("env", "gridworld").strip().lower()

    if env_name == "gridworld":
        tasks = _build_gridworld_tasks(cfg)
    elif env_name == "reacher":
        tasks = _build_reacher_tasks(cfg)
    elif env_name == "maze":
        tasks = _build_maze_tasks(cfg)
        cells = [t.goal_cells[t.task_index] for t in tasks]
        logger.info(f"MAZE tasks created with goal cells: {cells}")
    else:
        raise ValueError(
            f"Unsupported TASK.env '{env_name}'. Supported values: gridworld, reacher, maze"
        )

    return tasks

class AvgFGSFDQN(FGSFDQN):
    """Algorithm 1 with averaging over n_averaging next-state samples per pivot.

    All update logic lives in FGSFDQN (algorithm='alg1_averaged'); with n_averaging == 1
    this reduces to Algorithm 1.
    """
    pass

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=1e-3, device=None):
        super(MLP, self).__init__()
        if device is None:
             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
             self.device = device
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.network(x.to(self.device))


def run_single_trial(agent_name, agent_params, cfg, tasks, trial_index, override_params=None, display_name=None):
    """Train one trial and return reward history."""
    if override_params is None:
        override_params = {}

    display = display_name if display_name is not None else agent_name

    if not tasks:
        raise ValueError("run_single_trial called with empty tasks list")
    logger.debug(f"run_single_trial starting with {len(tasks)} tasks, first task={tasks[0]}")

    wandb_run = init_wandb_run(display, trial_index, cfg)

    base_seed = int(cfg["GENERAL"].get("seed", 0))
    set_global_seed(base_seed + trial_index)

    logger.info(f"[{display:<10s} | T{trial_index + 1}] Starting trial (seed={base_seed + trial_index})")

    n_samples = int(cfg["GENERAL"]["n_samples"])
    save_ev = int(cfg["AGENT"]["save_ev"])
    print_ev = int(cfg["AGENT"].get("print_ev", 2000))

    if save_ev > n_samples:
        logger.warning(
            "save_ev (%d) is greater than n_samples (%d); "
            "agents will never append reward history during training."
            " trial_history may end up empty."
            % (save_ev, n_samples)
        )
    total_data_points = (n_samples * len(tasks)) // save_ev

    input_dim = tasks[0].encode_dim()
    n_actions = tasks[0].action_count()
    n_features = tasks[0].feature_dim()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer_size = int(cfg["GENERAL"]["buffer_size"])
    n_batch = int(cfg["GENERAL"]["n_batch"])
    gamma = float(cfg["AGENT"]["gamma"])
    epsilon = float(cfg["AGENT"]["epsilon"])
    T = int(cfg["AGENT"]["T"])

    algo = agent_params.get("algorithm", "alg1")
    if "n_averaging" in override_params:
        n_avg = override_params["n_averaging"]
    else:
        n_avg = int(cfg["FGSF"]["n_averaging"])
    # Averaging in continuous domains: sample the n_avg next states of each (s, a) from the simulator
    # (task.n_resample extra ones per transition), and grow the buffer so it keeps as many real transitions.
    averaged = agent_name in ["AvgFGSFDQN", "AVGFGSFDQN_LINEAR", "AVGFGSFDQN_SFR"] or algo in ["alg3", "alg4"]
    if averaged and n_avg > 1 and cfg.has_section("FGSF") and cfg["FGSF"].getboolean("simulator_resampling", fallback=False):
        if not all(hasattr(t, "n_resample") for t in tasks):
            raise ValueError("FGSF.simulator_resampling is only supported for the Reacher and PointMaze tasks")
        for t in tasks:
            t.n_resample = n_avg - 1
        buffer_size *= n_avg

    reward_model = override_params.get("reward_model", cfg["SFQL"].get("reward_model", "linear"))
    successor_representation = override_params.get(
        "successor_representation", cfg["SFQL"].get("successor_representation", "sf")
    )
    reward_hidden_units = int(override_params.get(
        "reward_hidden_units", cfg["SFQL"].getint("reward_hidden_units", fallback=128)
    ))
    reward_learning_rate = float(override_params.get(
        "reward_learning_rate", cfg["SFQL"].getfloat("reward_learning_rate", fallback=1e-3)
    ))
    sfr_max_centers = int(override_params.get(
        "sfr_max_centers", cfg["SFQL"].getint("sfr_max_centers", fallback=64)
    ))
    sfr_center_threshold = float(override_params.get(
        "sfr_center_threshold", cfg["SFQL"].getfloat("sfr_center_threshold", fallback=0.0)
    ))
    sf_dim = sfr_max_centers if successor_representation == "sfr" else n_features

    # DQN/FG-DQN learning rate: [DQN] learning_rate if given, else [QL] learning_rate
    dqn_lr = cfg.getfloat("DQN", "learning_rate", fallback=float(cfg["QL"]["learning_rate"]))

    agent = None

    if agent_name == "DQN":
        def model_builder():
            return MLP(input_dim, n_actions, learning_rate=dqn_lr, device=device)

        buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = DQN(
            model_builder, buffer, gamma=gamma, epsilon=epsilon, T=T,
            encoding=tasks[0].encode, save_ev=save_ev, print_ev=print_ev
        )

    elif agent_name == "FGDQN":
        def model_builder():
            return MLP(input_dim, n_actions, learning_rate=dqn_lr, device=device)

        buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = FGDQN(
            model_builder, buffer, gamma=gamma, epsilon=epsilon, T=T,
            encoding=tasks[0].encode, save_ev=save_ev, print_ev=print_ev
        )

    elif agent_name in ["SFDQN", "SFDQN_LINEAR", "SFDQN_SFR"]:
        sf = DeepSF(
            input_dim=input_dim, n_actions=n_actions, n_features=sf_dim,
            learning_rate=float(cfg["SFQL"]["learning_rate"]),
            learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
            device=device,
            use_true_reward=cfg["SFQL"].getboolean("use_true_reward"),
            reward_model=reward_model,
            reward_hidden_units=reward_hidden_units,
            reward_learning_rate=reward_learning_rate,
            successor_representation=successor_representation,
            reward_input_dim=n_features,
            sfr_center_threshold=sfr_center_threshold,
        )
        buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = SFDQN(sf, buffer, gamma=gamma, epsilon=epsilon, T=T, encoding=tasks[0].encode,
                      save_ev=save_ev, print_ev=print_ev)

    elif agent_name in ["AvgFGSFDQN", "AVGFGSFDQN_LINEAR", "AVGFGSFDQN_SFR"]:
        sf = DeepFGSF(
            input_dim=input_dim, n_actions=n_actions, n_features=sf_dim,
            learning_rate=float(cfg["SFQL"]["learning_rate"]),
            learning_rate_prior=float(cfg["SFQL"].get("learning_rate_prior", cfg["SFQL"]["learning_rate"])),
            learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
            device=device,
            use_true_reward=cfg["SFQL"].getboolean("use_true_reward"),
            reward_model=reward_model,
            reward_hidden_units=reward_hidden_units,
            reward_learning_rate=reward_learning_rate,
            successor_representation=successor_representation,
            reward_input_dim=n_features,
            sfr_center_threshold=sfr_center_threshold,
        )
        buffer = ConditionalReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = AvgFGSFDQN(sf, buffer, gamma=gamma, T=T, epsilon=epsilon, encoding="task",
                           algorithm="alg1_averaged", n_averaging=n_avg, save_ev=save_ev, print_ev=print_ev,
                           pivot_batch_size=cfg["FGSF"].getint("pivot_batch_size", fallback=None))

    else:
        sf = DeepFGSF(
            input_dim=input_dim, n_actions=n_actions, n_features=sf_dim,
            learning_rate=float(cfg["SFQL"]["learning_rate"]),
            learning_rate_prior=float(cfg["SFQL"].get("learning_rate_prior", cfg["SFQL"]["learning_rate"])),
            learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
            device=device,
            use_true_reward=cfg["SFQL"].getboolean("use_true_reward"),
            reward_model=reward_model,
            reward_hidden_units=reward_hidden_units,
            reward_learning_rate=reward_learning_rate,
            successor_representation=successor_representation,
            reward_input_dim=n_features,
            sfr_center_threshold=sfr_center_threshold,
        )
        if algo in ["alg3", "alg4"]:
            buffer = ConditionalReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        else:
            buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = FGSFDQN(sf, buffer, gamma=gamma, T=T, epsilon=epsilon, encoding="task",
                        algorithm=algo, n_averaging=n_avg, save_ev=save_ev, print_ev=print_ev,
                        pivot_batch_size=cfg["FGSF"].getint("pivot_batch_size", fallback=None))

    agent.reset()
    if hasattr(agent, "sf"):
        # SF network width and gradient clipping (defaults: 256x256, clip at norm 1.0)
        agent.sf.hidden_dims = [int(x) for x in cfg["SFQL"].get("sf_hidden", "256,256").split(",")]
        agent.sf.grad_clip = cfg["SFQL"].getfloat("grad_clip", fallback=1.0)
        agent.sf.target_update_ev = cfg["SFQL"].getint("target_update_ev", fallback=1000)  # SFDQN target net
        # FG update variants (defaults = paper Alg. 1): prior_state s|next, update_mode joint|grouped
        agent.sf.prior_state = cfg["FGSF"].get("prior_state", "s")
        agent.sf.update_mode = cfg["FGSF"].get("update_mode", "joint")
        # paper Alg. 1 initializes the SFs of a new task from the previous task's SFs
        agent.copy_previous_sf = cfg["SFQL"].getboolean("copy_previous_sf", fallback=False)
    agent.agent_name = display
    agent.trial_index = trial_index
    agent.total_tasks = len(tasks)
    agent.n_samples = n_samples
    agent.print_ev = print_ev
    agent.progress_logger = logger.info

    if wandb_run is not None:
        wandb_step = 0

        def _log_reward(step, cumulative_reward, task_index=None):
            nonlocal wandb_step
            task_index = -1 if task_index is None else task_index
            wandb_run.log(
                {
                    "cumulative_reward": float(cumulative_reward),
                    "agent": display,
                    "trial": trial_index + 1,
                    "task_index": task_index,
                    f"cumulative_reward/task_{task_index}": float(cumulative_reward),
                },
                step=wandb_step,
            )
            wandb_step += 1

        agent.reward_log_callback = _log_reward

        def _log_diagnostics(entry):
            # logged at the current reward step so wandb steps stay monotone
            wandb_run.log({f"diag/{k}": v for k, v in entry.items()
                           if isinstance(v, (int, float)) and k != "step"}, step=max(wandb_step - 1, 0))

        agent.diag_log_callback = _log_diagnostics

    trial_history = []

    if isinstance(agent, FGSFDQN) and algo in ["alg2", "alg3"]:
        # alg2/alg3 use randomized task sampling.
        n_total_steps = n_samples * len(tasks)
        agent.train_randomized(tasks, n_total_steps=n_total_steps)
        trial_history = list(agent.cum_reward_hist)
        if len(trial_history) == 0:
            trial_history = [agent.cum_reward]
        logger.info(
            f"[{display:<10s} | T{trial_index+1}] ✓ Randomized training completed "
            f"| Total Reward: {agent.cum_reward:.2f}"
        )
    else:
        task_totals = []
        for t_idx, task in enumerate(tasks):
            agent.cum_reward = 0
            agent.train_on_task(task, n_samples=n_samples)
            task_totals.append(float(agent.cum_reward))

            points_per_task = n_samples // save_ev
            curr_hist = agent.cum_reward_hist

            # Keep at least one value when history is empty.
            if len(curr_hist) == 0:
                curr_hist = [agent.cum_reward]

            if len(curr_hist) >= points_per_task and points_per_task > 0:
                task_data = curr_hist[-points_per_task:]
            else:
                task_data = curr_hist
            trial_history.extend(task_data)
            logger.info(
                f"[{display:<10s} | T{trial_index+1}] ✓ Completed Task {t_idx+1}/{len(tasks)} "
                f"| Task Reward: {agent.cum_reward:.2f}"
            )

    if len(trial_history) != total_data_points:
        if len(trial_history) < total_data_points:
            if trial_history:
                trial_history.extend([trial_history[-1]] * (total_data_points - len(trial_history)))
            else:
                trial_history = [0.0] * total_data_points
        else:
            trial_history = trial_history[:total_data_points]

    diag_hist = getattr(agent, "diag_hist", None)
    if diag_hist:
        diag_dir = os.path.join(cfg["GENERAL"].get("weights_dir", "updated_weights"), "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        with open(os.path.join(diag_dir, f"{display}_trial_{trial_index}.json"), "w") as f:
            json.dump(diag_hist, f, indent=1)

    if cfg["AGENT"].getboolean("save_checkpoints", fallback=True):
        save_agent_weights(agent, agent_name, trial_index,
                           root=cfg["GENERAL"].get("weights_dir", "updated_weights"))

    if isinstance(agent, FGSFDQN) and algo in ["alg2", "alg3"]:
        total_trial_rew = float(agent.cum_reward)
    else:
        total_trial_rew = float(sum(task_totals))
    logger.info(
        f"[{display:<10s} | T{trial_index+1}] ★ Trial finished successfully "
        f"| Cumulative Reward: {total_trial_rew:.2f}"
    )

    if wandb_run is not None:
        wandb_run.finish()

    return np.array(trial_history)


def run_experiment(agent_name, agent_params, cfg, tasks, n_trials, override_params=None, display_name=None):
    display = display_name or agent_name
    all_trials_data = []
    for trial in range(n_trials):
        history = run_single_trial(agent_name, agent_params, cfg, tasks, trial, override_params, display_name=display)
        all_trials_data.append(history)
    logger.info(f"[{display:<10s}] All {n_trials} trial(s) completed successfully")
    return np.array(all_trials_data)


def _print_experiment_summary(raw_results, n_trials, total_duration, n_tasks=1, points_per_task=None,
                              randomized=()):
    """Prints a clean summary table of results across all methods."""
    table_lines = [
        "\n" + "=" * 96,
        "  EXPERIMENT EXECUTION SUMMARY",
        "=" * 96,
        f"  Total Duration: {total_duration:.1f}s ({total_duration/60.0:.2f} min)",
        "-" * 96,
        f"  {'Method':<16s} {'Trials':<8s} {'Mean Total Rew':<16s} {'Std Total':<12s} {'Task Breakdown (Mean Rew)':<40s}",
        "-" * 96,
    ]

    for method_name, trials_arr in raw_results.items():
        if trials_arr is None or len(trials_arr) == 0:
            table_lines.append(f"  {method_name:<16s} {'0/' + str(n_trials):<8s} {'N/A':<16s} {'N/A':<12s}")
            continue

        arr = np.asarray(trials_arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        valid_count = arr.shape[0]
        if valid_count == 0 or arr.shape[1] == 0:
            table_lines.append(f"  {method_name:<16s} {'0/' + str(n_trials):<8s} {'N/A':<16s} {'N/A':<12s}")
            continue

        # Randomized-task methods (alg2/alg3) log one running total over all tasks, not per-task totals.
        if method_name in randomized:
            total_rewards = arr[:, -1]
            breakdown_str = f"Final (all tasks, randomized): {np.mean(total_rewards):.1f}"
        elif points_per_task and points_per_task > 0 and n_tasks > 1 and arr.shape[1] >= points_per_task * n_tasks:
            # Multi-task sequential training: extract each task's cumulative reward
            task_ends_matrix = []
            for trial_idx in range(valid_count):
                ends = [arr[trial_idx, (t + 1) * points_per_task - 1] for t in range(n_tasks)]
                task_ends_matrix.append(ends)
            task_ends_matrix = np.array(task_ends_matrix)
            total_rewards = np.sum(task_ends_matrix, axis=1)
            mean_task_rewards = np.mean(task_ends_matrix, axis=0)
            breakdown_str = " | ".join([f"T{t+1}: {r:.1f}" for t, r in enumerate(mean_task_rewards)])
        else:
            total_rewards = arr[:, -1]
            breakdown_str = f"Final: {np.mean(total_rewards):.1f}"

        mean_total = float(np.mean(total_rewards))
        std_total = float(np.std(total_rewards))
        status_trials = f"{valid_count}/{n_trials}"

        table_lines.append(
            f"  {method_name:<16s} {status_trials:<8s} {mean_total:>14.2f} {std_total:>12.2f}   {breakdown_str}"
        )

    table_lines.append("=" * 96 + "\n")
    summary_text = "\n".join(table_lines)
    print(summary_text, flush=True)
    logger.info("Experiment summary generated.")


# display name -> (agent name, agent params, override params)
JOB_REGISTRY = {
    "DQN": ("DQN", {}, None),
    "FGDQN": ("FGDQN", {}, None),
    "SFDQN": ("SFDQN", {}, {"reward_model": "linear", "successor_representation": "sf"}),
    "FG-SFDQN": ("FGSFDQN", {"algorithm": "alg1"},
                 {"n_averaging": 1, "reward_model": "linear", "successor_representation": "sf"}),
    "AvgFGSFDQN": ("AvgFGSFDQN", {"algorithm": "alg1_averaged"},
                   {"n_averaging": 5, "reward_model": "linear", "successor_representation": "sf"}),
    "AvgFGSFDQN-N10": ("AvgFGSFDQN", {"algorithm": "alg1_averaged"},
                       {"n_averaging": 10, "reward_model": "linear", "successor_representation": "sf"}),
    "AvgFGSFDQN-N20": ("AvgFGSFDQN", {"algorithm": "alg1_averaged"},
                       {"n_averaging": 20, "reward_model": "linear", "successor_representation": "sf"}),
    "alg2": ("alg2", {"algorithm": "alg2"}, {"n_averaging": 1}),
    "alg3": ("alg3", {"algorithm": "alg3"}, {"n_averaging": 5}),
}
# Without --methods, every entry of JOB_REGISTRY is run.
DEFAULT_METHODS = list(JOB_REGISTRY)


def run_all_experiments(cfg, tasks, n_trials=5):
    """Run all configured agents, optionally in parallel."""
    if not tasks:
        raise ValueError("Task list is empty – cannot run experiments")

    env_name = cfg["TASK"].get("env", "gridworld").strip()
    n_samples = int(cfg["GENERAL"].get("n_samples", 15000))
    print_ev = int(cfg["AGENT"].get("print_ev", 2000))

    selected = [m.strip() for m in cfg["GENERAL"].get("methods", ",".join(DEFAULT_METHODS)).split(",") if m.strip()]
    unknown = [m for m in selected if m not in JOB_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}; available: {list(JOB_REGISTRY)}")
    jobs = [(name, *JOB_REGISTRY[name][:2], cfg, tasks, n_trials, JOB_REGISTRY[name][2]) for name in selected]
    randomized = {name for name in selected if JOB_REGISTRY[name][1].get("algorithm") in ("alg2", "alg3")}

    raw_ids = cfg["GENERAL"].get("trial_ids", "").strip()
    trial_ids = [int(x) for x in raw_ids.split(",") if x.strip()] if raw_ids else list(range(n_trials))
    n_trials = len(trial_ids)
    requested_workers = int(cfg["GENERAL"].get("n_workers", 1))
    n_jobs = len(jobs) * n_trials
    n_workers = max(1, min(requested_workers, n_jobs))
    job_names = [j[0] for j in jobs]

    banner = (
        "\n" + "=" * 92 + "\n"
        "  PARALLEL REINFORCEMENT LEARNING EXPERIMENTS\n"
        + "=" * 92 + "\n"
        f"  Environment   : {env_name} ({len(tasks)} task(s): {[t.__class__.__name__ for t in tasks]})\n"
        f"  Active Methods: {', '.join(job_names)}\n"
        f"  Total Jobs    : {n_jobs} ({len(jobs)} method(s) × {n_trials} trial(s))\n"
        f"  Steps / Task  : {n_samples:,} (Total steps / trial: {n_samples * len(tasks):,})\n"
        f"  Print Every   : {print_ev:,} steps\n"
        f"  Worker Pool   : {n_workers} process(es) (multiprocessing spawn)\n"
        f"  Log File      : {LOG_FILE}\n"
        + "=" * 92
    )
    print(banner, flush=True)
    logger.info(f"Initialized experiment suite with {len(jobs)} methods across {n_trials} trials on {n_workers} workers.")

    import time
    start_time = time.time()
    raw = {}

    failures = []
    if n_workers > 1:
        # Use spawn for safer multiprocessing with torch.
        ctx = mp.get_context('spawn')
        raw = {display: [None] * n_trials for (display, *_rest) in jobs}

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            future_to_key = {}
            for display, a_name, a_params, c, t, trials, overrides in jobs:
                for pos, trial_idx in enumerate(trial_ids):
                    fut = executor.submit(run_single_trial, a_name, a_params, c, t, trial_idx, overrides, display)
                    future_to_key[fut] = (display, trial_idx, pos, time.time())

            completed = 0
            for future in as_completed(future_to_key):
                display, trial_idx, pos, job_start = future_to_key[future]
                job_duration = time.time() - job_start
                completed += 1
                try:
                    history = future.result()
                    raw[display][pos] = history
                    if len(history) > 0:
                        save_ev = int(cfg["AGENT"].get("save_ev", 200))
                        ppt = n_samples // save_ev if save_ev > 0 else 0
                        if display in randomized:
                            total_rew = float(history[-1])
                        elif ppt > 0 and len(tasks) > 1 and len(history) >= ppt * len(tasks):
                            total_rew = float(sum(history[(t + 1) * ppt - 1] for t in range(len(tasks))))
                        else:
                            total_rew = float(history[-1])
                    else:
                        total_rew = 0.0
                    logger.info(
                        f"[{display:<10s} | T{trial_idx+1}] ★ Job finished in {job_duration:.1f}s "
                        f"| Total Reward: {total_rew:.2f} ({completed}/{n_jobs} jobs done)"
                    )
                except Exception as exc:
                    tb = traceback.format_exc()
                    logger.error(f"[{display:<10s} | T{trial_idx+1}] ✗ FAILED after {job_duration:.1f}s: {exc}\n{tb}")
                    failures.append((display, trial_idx))

        for display in list(raw.keys()):
            trials_list = raw[display]
            target_len = None
            for h in trials_list:
                if h is not None:
                    target_len = len(h)
                    break
            for idx, h in enumerate(trials_list):
                if h is None:
                    logger.warning(f"[{display} | T{idx+1}] returned None; inserting zeros")
                    trials_list[idx] = np.zeros(target_len if target_len is not None else 0)
            raw[display] = np.stack(trials_list, axis=0)
    else:
        for job in jobs:
            display, a_name, a_params, c, t, trials, overrides = job
            all_trials_data = []
            for trial_idx in trial_ids:
                h = run_single_trial(a_name, a_params, c, t, trial_idx, overrides, display)
                all_trials_data.append(h)
            raw[display] = np.stack(all_trials_data, axis=0)

    total_duration = time.time() - start_time
    save_ev = int(cfg["AGENT"].get("save_ev", 200))
    points_per_task = n_samples // save_ev if save_ev > 0 else 0
    _print_experiment_summary(raw, n_trials, total_duration, n_tasks=len(tasks), points_per_task=points_per_task,
                              randomized=randomized)

    if failures:
        raise RuntimeError(f"{len(failures)} job(s) failed: {failures}")
    return raw

def save_raw_results(raw_results, filename="raw_results.npz"):
    # Store numeric arrays when possible; keep object fallback.
    clean = {}
    for k, v in raw_results.items():
        try:
            clean[k] = np.asarray(v, dtype=float)
        except Exception:
            logger.warning(f"save_raw_results: could not convert '{k}' to float array;"
                           " saving with object dtype")
            clean[k] = np.asarray(v, dtype=object)
    np.savez_compressed(filename, **clean)
    print(f"[OK] Raw results saved to {filename}")

def save_tasks(tasks, cfg, root="updated_weights"):
    task_dir = os.path.join(root, "tasks")
    os.makedirs(task_dir, exist_ok=True)

    try:
        with open(os.path.join(task_dir, "tasks.pkl"), "wb") as f:
            pickle.dump(tasks, f)
    except Exception as e:
        logger.warning(f"Could not pickle tasks list (likely due to live env handles): {e}")

    env_name = cfg["TASK"].get("env", "gridworld").strip().lower()

    meta = {
        "env": env_name,
        "n_tasks": len(tasks),
        "weights": [np.asarray(t.get_w()).tolist() for t in tasks],
    }

    if env_name == "gridworld":
        meta["maze"] = ast.literal_eval(cfg["TASK"]["maze"])
        meta["shape_rewards"] = [dict(t.shape_rewards) for t in tasks]
    elif env_name == "reacher":
        meta["target_positions"] = _parse_tuple_list(
            cfg["TASK"].get("reacher_train_targets", "[(0.14, 0.0), (-0.14, 0.0), (0.0, 0.14), (0.0, -0.14)]")
        )
        meta["include_target_in_state"] = cfg["TASK"].getboolean("reacher_include_target_in_state", fallback=False)
        meta["continuous_actions"] = cfg["TASK"].getboolean("continuous_actions", fallback=True)
    elif env_name == "maze":
        meta["env_id"] = cfg["TASK"].get("maze_env_id", "PointMaze_Large-v3")
        meta["goal_cells"] = [tuple(map(int, c)) for c in ast.literal_eval(cfg["TASK"].get("maze_goal_cells", "[(1, 1), (1, 3), (3, 3)]"))]
        meta["effective_goal_cells_per_task"] = [
            tuple(map(int, t.goal_cells[t.task_index])) for t in tasks
        ]
        meta["effective_goal_xys_per_task"] = [
            np.asarray(t.goal_xys[t.task_index], dtype=float).tolist() for t in tasks
        ]
        reset_cell_raw = cfg["TASK"].get("maze_reset_cell", fallback=None)
        meta["reset_cell"] = tuple(map(int, ast.literal_eval(reset_cell_raw))) if reset_cell_raw else None
        meta["include_goal_in_state"] = cfg["TASK"].getboolean("maze_include_goal_in_state", fallback=True)
        meta["continuous_actions"] = cfg["TASK"].getboolean("continuous_actions", fallback=True)
        meta["goal_jitter"] = cfg["TASK"].getfloat("maze_goal_jitter", fallback=0.0)

    # write atomically: several jobs may share one weights directory
    tmp_path = os.path.join(task_dir, f"tasks_meta.json.tmp{os.getpid()}")
    with open(tmp_path, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_path, os.path.join(task_dir, "tasks_meta.json"))

    logger.info("Tasks saved successfully")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.cfg", help="config file to use")
    parser.add_argument("--methods", default=None,
                        help="comma-separated methods, e.g. DQN,SFDQN,FG-SFDQN,AvgFGSFDQN")
    parser.add_argument("--out", default="updated_results.npz", help="raw results file")
    parser.add_argument("--weights-dir", default=None, help="checkpoint directory (default: updated_weights)")
    parser.add_argument("--trial-ids", default=None, help="comma-separated trial indices to run (default: all)")
    parser.add_argument("--wandb-project", default=None, help="wandb project name")
    parser.add_argument("--wandb-group", default=None, help="wandb group (e.g. experiment name)")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    with open(LOG_FILE, "w") as f:
        pass
        
    logger.info("==========================================")
    logger.info("Experiment started")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("==========================================")

    cfg = load_config(args.config)
    if args.methods is not None:
        cfg["GENERAL"]["methods"] = args.methods
    logger.info(f"Config: {args.config}")
    if args.weights_dir is not None:
        cfg["GENERAL"]["weights_dir"] = args.weights_dir
    for key, val in (("trial_ids", args.trial_ids), ("wandb_project", args.wandb_project),
                     ("wandb_group", args.wandb_group)):
        if val is not None:
            cfg["GENERAL"][key] = val

    base_seed = int(cfg["GENERAL"].get("seed", 0))
    set_global_seed(base_seed)

    tasks = build_task_sequence(cfg)
    
    n_trials = int(cfg["GENERAL"]["n_trials"])
    
    raw_results = run_all_experiments(cfg, tasks, n_trials=n_trials)
    
    try:
        save_tasks(tasks, cfg, root=cfg["GENERAL"].get("weights_dir", "updated_weights"))
    except Exception as e:
        print(f"Failed to save tasks: {e}")
        
    save_raw_results(raw_results, filename=args.out)