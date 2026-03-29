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

LOG_FILE = "logger.log"

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


def _get_wandb_run_name(agent_name, trial_index):
    return f"{agent_name}_trial_{trial_index+1}"


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
            project="FG-SFDQN",
            name=_get_wandb_run_name(agent_name, trial_index),
            reinit=True,
            config=_wandb_config_from_cfg(cfg),
            save_code=False,
        )
        return run
    except Exception as exc:
        logger.warning(f"wandb.init failed: {exc}")
        return None


CONFIG_CONTENT = """
[GENERAL]
seed=0
n_samples=30000
n_tasks=8
n_trials=3
n_batch=512
buffer_size=200000
n_workers=32

[TASK]
env=gridworld
# Standard Barreto et al. Gridworld layout
maze=[
    ['1', ' ', ' ', ' ', ' ', '2', 'X', ' ', ' ', ' ', ' ', ' ', 'G'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '1', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['2', ' ', ' ', ' ', ' ', '3', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['X', 'X', '3', ' ', 'X', 'X', 'X', 'X', 'X', ' ', '1', 'X', 'X'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', '2', ' ', ' ', ' ', ' ', '3'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '2', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['_', ' ', ' ', ' ', ' ', ' ', 'X', '3', ' ', ' ', ' ', ' ', '1']]
reacher_train_targets=[(0.14, 0.0), (-0.14, 0.0), (0.0, 0.14), (0.0, -0.14)]
reacher_include_target_in_state=False
maze_env_id=PointMaze_UMaze-v3
maze_goal_cells=[(1, 1)]
maze_reset_cell=(3, 3)
maze_include_goal_in_state=True
continuous_actions=True

maze_similarity_radius=1
maze_goal_jitter=0

[AGENT]
gamma=0.95
epsilon=0.55
T=200
print_ev=2000
save_ev=200
save_checkpoints=True

[SFQL]
learning_rate=0.001
learning_rate_prior=0.00001
learning_rate_w=0.5
reward_learning_rate=0.001
use_true_reward=False
hidden_units=128
reward_model=linear
reward_hidden_units=128
successor_representation=sf
sfr_max_centers=16
sfr_center_threshold=0.05

[QL]
learning_rate=0.5

[FGSF]
n_averaging=5
"""

def setup_configs():
    if not os.path.exists("configs"):
        os.makedirs("configs")
    with open("configs/config.cfg", "w") as f:
        f.write(CONFIG_CONTENT)

def load_config(path="configs/config.cfg"):
    cfg = configparser.ConfigParser()
    cfg.read(path)
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
    tasks = []
    for i in range(n_tasks):
        rewards = rewards_pool[i % len(rewards_pool)]
        tasks.append(Shapes(maze=maze, shape_rewards=rewards))
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
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        phi = self.phi(s, a, s1)
        if isinstance(phi, torch.Tensor):
            phi = phi.detach().cpu().numpy()
        phi_target = self.sf.encode_transition_feature(phi)
        self.sf.update_reward(phi, r, self.task_index)
    
        self.buffer.append(s_enc, a, phi_target, s1_enc, gamma)
    
        if self.algorithm not in ['alg1', 'alg1_averaged']:
            return
    
        if self.n_averaging > 1:
            try:
                p_s = np.asarray(s_enc).reshape(1, -1)
            except Exception:
                p_s = np.array([s_enc]).reshape(1, -1)
    
            if not hasattr(self.buffer, "sample_conditional"):
                batch = self.buffer.replay()
                if batch:
                    self._update_batch_grouped_by_prior(batch, self.task_index)
                return
    
            cond_batch = self.buffer.sample_conditional(p_s, a, self.n_averaging)
    
            if cond_batch:
                _, _, _, c_next_states, _ = cond_batch
    
                c_next_states = np.asarray(c_next_states)
                if c_next_states.ndim == 3 and c_next_states.shape[1] == 1:
                    c_next_states = c_next_states.squeeze(1)
                elif c_next_states.ndim == 1:
                    c_next_states = c_next_states.reshape(1, -1)
    
                mean_next_state_feature = np.mean(c_next_states, axis=0)
                c = self._get_gpi_policy(mean_next_state_feature, self.task_index)[0]
                task_c = c if c != self.task_index else None
    
                self.sf.update_averaged(p_s, a, cond_batch, self.task_index, task_c)
                return
    
        batch = self.buffer.replay()
        if batch:
            self._update_batch_grouped_by_prior(batch, self.task_index)

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


def run_single_trial(agent_name, agent_params, cfg, tasks, trial_index, override_params=None):
    """Train one trial and return reward history."""
    if override_params is None:
        override_params = {}

    if not tasks:
        raise ValueError("run_single_trial called with empty tasks list")
    logger.debug(f"run_single_trial starting with {len(tasks)} tasks, first task={tasks[0]}")

    print(f"  {agent_name} | Trial {trial_index + 1}")
    logger.info(f"{agent_name} Trial {trial_index + 1} started")

    wandb_run = init_wandb_run(agent_name, trial_index, cfg)

    base_seed = int(cfg["GENERAL"].get("seed", 0))
    set_global_seed(base_seed + trial_index)

    n_samples = int(cfg["GENERAL"]["n_samples"])
    save_ev = int(cfg["AGENT"]["save_ev"])
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

    agent = None

    if agent_name == "DQN":
        def model_builder():
            return MLP(input_dim, n_actions, learning_rate=float(cfg["QL"]["learning_rate"]), device=device)

        buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = DQN(
            model_builder, buffer, gamma=gamma, epsilon=epsilon, T=T,
            encoding=tasks[0].encode, save_ev=save_ev
        )

    elif agent_name == "FGDQN":
        def model_builder():
            return MLP(input_dim, n_actions, learning_rate=float(cfg["QL"]["learning_rate"]), device=device)

        buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = FGDQN(
            model_builder, buffer, gamma=gamma, epsilon=epsilon, T=T,
            encoding=tasks[0].encode, save_ev=save_ev
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
        agent = SFDQN(sf, buffer, gamma=gamma, epsilon=epsilon, T=T, encoding=tasks[0].encode, save_ev=save_ev)

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
                           algorithm="alg1_averaged", n_averaging=n_avg, save_ev=save_ev)

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
        if algo in ["alg3"]:
            buffer = ConditionalReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        else:
            buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = FGSFDQN(sf, buffer, gamma=gamma, T=T, epsilon=epsilon, encoding="task",
                        algorithm=algo, n_averaging=n_avg, save_ev=save_ev)

    agent.reset()
    if wandb_run is not None:
        wandb_step = 0

        def _log_reward(step, cumulative_reward, task_index=None):
            nonlocal wandb_step
            task_index = -1 if task_index is None else task_index
            wandb_run.log(
                {
                    "cumulative_reward": float(cumulative_reward),
                    "agent": agent_name,
                    "trial": trial_index + 1,
                    "task_index": task_index,
                    f"cumulative_reward/task_{task_index}": float(cumulative_reward),
                },
                step=wandb_step,
            )
            wandb_step += 1

        agent.reward_log_callback = _log_reward

    trial_history = []

    if isinstance(agent, FGSFDQN) and algo in ["alg2", "alg3"]:
        # alg2/alg3 use randomized task sampling.
        n_total_steps = n_samples * len(tasks)
        agent.train_randomized(tasks, n_total_steps=n_total_steps)
        trial_history = list(agent.cum_reward_hist)
        if len(trial_history) == 0:
            trial_history = [agent.cum_reward]
        logger.info(
            f"Agent={agent_name} | Trial={trial_index+1} | "
            f"Randomized training completed | Cumulative Reward={agent.cum_reward:.2f}"
        )
    else:
        for t_idx, task in enumerate(tasks):
            agent.cum_reward = 0
            agent.train_on_task(task, n_samples=n_samples)

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
                f"Agent={agent_name} | Trial={trial_index+1} | "
                f"Task={t_idx+1}/{len(tasks)} | "
                f"Cumulative Reward={agent.cum_reward:.2f}"
            )

    if len(trial_history) != total_data_points:
        if len(trial_history) < total_data_points:
            if trial_history:
                trial_history.extend([trial_history[-1]] * (total_data_points - len(trial_history)))
            else:
                trial_history = [0.0] * total_data_points
        else:
            trial_history = trial_history[:total_data_points]

    if cfg["AGENT"].getboolean("save_checkpoints", fallback=True):
        save_agent_weights(agent, agent_name, trial_index, root="updated_weights")

    if wandb_run is not None:
        wandb_run.finish()

    return np.array(trial_history)


def run_experiment(agent_name, agent_params, cfg, tasks, n_trials, override_params=None):
    all_trials_data = []
    for trial in range(n_trials):
        history = run_single_trial(agent_name, agent_params, cfg, tasks, trial, override_params)
        all_trials_data.append(history)
    logger.info(f"All experiments for {agent_name} completed successfully")
    return np.array(all_trials_data)


def run_all_experiments(cfg, tasks, n_trials=5):
    """Run all configured agents, optionally in parallel."""
    logger.info("==========================================")
    logger.info("RUNNING ALL EXPERIMENTS (PARALLEL)")
    logger.info("==========================================")

    if not tasks:
        raise ValueError("Task list is empty – cannot run experiments")
    logger.info(f"Running experiments on {len(tasks)} tasks: {[t.__class__.__name__ for t in tasks]}")

    jobs = [
        ("DQN", "DQN", {}, cfg, tasks, n_trials, None),
        ("SFDQN", "SFDQN", {}, cfg, tasks, n_trials,
         {"reward_model": "linear", "successor_representation": "sf"}),
        # ("SFR-Nonlinear", "SFDQN_SFR", {}, cfg, tasks, n_trials,
        #  {
        #      "reward_model": "nonlinear",
        #      "successor_representation": "sfr",
        #      "reward_learning_rate": 0.03,
        #      "sfr_max_centers": 4,
        #      "sfr_center_threshold": 0.05,
        #  }),
        ("FG-SFDQN", "FGSFDQN", {"algorithm": "alg1"}, cfg, tasks, n_trials,
         {"n_averaging": 1, "reward_model": "linear", "successor_representation": "sf"}),
        # ("FG-SFR-Nonlinear", "FGSFDQN_SFR", {"algorithm": "alg1"}, cfg, tasks, n_trials,
        #  {
        #      "n_averaging": 1,
        #      "reward_model": "nonlinear",
        #      "successor_representation": "sfr",
        #      "reward_learning_rate": 0.03,
        #      "sfr_max_centers": 4,
        #      "sfr_center_threshold": 0.05,
        #  }),
        ("alg2", "alg2", {"algorithm": "alg2"}, cfg, tasks, n_trials, {"n_averaging": 1}),
        ("alg3", "alg3", {"algorithm": "alg3"}, cfg, tasks, n_trials, {"n_averaging": 5}),
        ("FGDQN", "FGDQN", {}, cfg, tasks, n_trials, None),
    ]

    requested_workers = int(cfg["GENERAL"].get("n_workers", 1))
    n_jobs = len(jobs) * n_trials
    n_workers = max(1, min(requested_workers, n_jobs))
    if requested_workers != n_workers:
        logger.info(
            f"Adjusted n_workers from {requested_workers} to {n_workers} based on total jobs={n_jobs}"
        )
    raw = {}

    print(f"\n============================================")
    print(f"RUNNING EXPERIMENTS WITH {n_workers} WORKERS")
    print(f"============================================")

    if n_workers > 1:
        # Use spawn for safer multiprocessing with torch.
        ctx = mp.get_context('spawn')
        raw = {display: [None] * n_trials for (display, *_rest) in jobs}

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            future_to_key = {}
            for display, a_name, a_params, c, t, trials, overrides in jobs:
                for trial_idx in range(n_trials):
                    fut = executor.submit(run_single_trial, a_name, a_params, c, t, trial_idx, overrides)
                    future_to_key[fut] = (display, trial_idx)

            for future in as_completed(future_to_key):
                display, trial_idx = future_to_key[future]
                try:
                    history = future.result()
                    raw[display][trial_idx] = history
                    print(f"[SUCCESS] {display} trial {trial_idx+1} completed.")
                except Exception as exc:
                    tb = traceback.format_exc()
                    print(f"[ERROR] {display} trial {trial_idx+1} generated an exception: {exc}\n{tb}")
                    logger.error(f"{display} trial {trial_idx+1} failed: {exc}\n{tb}")

        for display in list(raw.keys()):
            trials_list = raw[display]
            target_len = None
            for h in trials_list:
                if h is not None:
                    target_len = len(h)
                    break
            for idx, h in enumerate(trials_list):
                if h is None:
                    logger.warning(f"{display} trial {idx+1} returned None; inserting zeros")
                    trials_list[idx] = np.zeros(target_len if target_len is not None else 0)
            raw[display] = np.stack(trials_list, axis=0)
    else:
        for job in jobs:
            name, a_name, a_params, c, t, trials, overrides = job
            raw[name] = run_experiment(a_name, a_params, c, t, trials, overrides)

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

    with open(os.path.join(task_dir, "tasks_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Tasks saved successfully")

if __name__ == '__main__':
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

    setup_configs()
    cfg = load_config()

    base_seed = int(cfg["GENERAL"].get("seed", 0))
    set_global_seed(base_seed)

    tasks = build_task_sequence(cfg)
    
    n_trials = int(cfg["GENERAL"]["n_trials"])
    
    raw_results = run_all_experiments(cfg, tasks, n_trials=n_trials)
    
    try:
        save_tasks(tasks, cfg)
    except Exception as e:
        print(f"Failed to save tasks: {e}")
        
    save_raw_results(raw_results, filename="updated_results.npz")