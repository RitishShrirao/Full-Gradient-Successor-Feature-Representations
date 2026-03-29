import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tasks.gridworld import Shapes
from agents.buffer import ReplayBuffer, ConditionalReplayBuffer
from agents.dqn import DQN
from agents.fgdqn import FGDQN
from agents.sfdqn import SFDQN
from agents.fgsfdqn import FGSFDQN
from features.deep import DeepSF
from features.deep_fg import DeepFGSF


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=1e-3, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.network(x.to(self.device))


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_gridworld_tasks(n_tasks: int):
    maze = np.array([
        ["1", " ", " ", " ", " ", "2", "X", " ", " ", " ", " ", " ", "G"],
        [" ", " ", " ", " ", " ", " ", "X", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "1", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "X", " ", " ", " ", " ", " ", " "],
        ["2", " ", " ", " ", " ", "3", "X", " ", " ", " ", " ", " ", " "],
        ["X", "X", "3", " ", "X", "X", "X", "X", "X", " ", "1", "X", "X"],
        [" ", " ", " ", " ", " ", " ", "X", "2", " ", " ", " ", " ", "3"],
        [" ", " ", " ", " ", " ", " ", "X", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "2", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "X", " ", " ", " ", " ", " ", " "],
        ["_", " ", " ", " ", " ", " ", "X", "3", " ", " ", " ", " ", "1"],
    ], dtype=str)

    rewards_pool = [
        {"1": 1.0, "2": 0.0, "3": 0.0},
        {"1": 0.0, "2": 1.0, "3": 0.0},
        {"1": 0.0, "2": 0.0, "3": 1.0},
        {"1": 1.0, "2": -1.0, "3": 0.0},
        {"1": 0.0, "2": 1.0, "3": -1.0},
    ]

    tasks = []
    for i in range(n_tasks):
        rewards = rewards_pool[i % len(rewards_pool)]
        tasks.append(Shapes(maze=maze, shape_rewards=rewards))
    return tasks


def init_agent(method_name: str, tasks, n_batch: int, buffer_size: int, device):
    input_dim = tasks[0].encode_dim()
    n_actions = tasks[0].action_count()
    n_features = tasks[0].feature_dim()

    gamma = 0.95
    epsilon = 0.55
    T = 200

    if method_name == "DQN":
        def model_builder():
            return MLP(input_dim, n_actions, learning_rate=0.5, device=device)

        buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = DQN(model_builder, buffer, gamma=gamma, epsilon=epsilon, T=T, encoding=tasks[0].encode)

    elif method_name == "FGDQN":
        def model_builder():
            return MLP(input_dim, n_actions, learning_rate=0.5, device=device)

        buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = FGDQN(model_builder, buffer, gamma=gamma, epsilon=epsilon, T=T, encoding=tasks[0].encode)

    elif method_name == "SFDQN":
        sf = DeepSF(
            input_dim=input_dim,
            n_actions=n_actions,
            n_features=n_features,
            learning_rate=1e-3,
            learning_rate_w=0.5,
            device=device,
            use_true_reward=False,
            reward_model="linear",
            successor_representation="sf",
            reward_input_dim=n_features,
            sfr_center_threshold=0.05,
        )
        buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = SFDQN(sf, buffer, gamma=gamma, epsilon=epsilon, T=T, encoding=tasks[0].encode)

    elif method_name == "FGSFDQN":
        sf = DeepFGSF(
            input_dim=input_dim,
            n_actions=n_actions,
            n_features=n_features,
            learning_rate=1e-3,
            learning_rate_prior=1e-5,
            learning_rate_w=0.5,
            device=device,
            use_true_reward=False,
            reward_model="linear",
            successor_representation="sf",
            reward_input_dim=n_features,
            sfr_center_threshold=0.05,
        )
        buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = FGSFDQN(
            sf,
            buffer,
            gamma=gamma,
            epsilon=epsilon,
            T=T,
            encoding="task",
            algorithm="alg1",
            n_averaging=1,
        )

    elif method_name == "alg2":
        sf = DeepFGSF(
            input_dim=input_dim,
            n_actions=n_actions,
            n_features=n_features,
            learning_rate=1e-3,
            learning_rate_prior=1e-5,
            learning_rate_w=0.5,
            device=device,
            use_true_reward=False,
            reward_model="linear",
            successor_representation="sf",
            reward_input_dim=n_features,
            sfr_center_threshold=0.05,
        )
        buffer = ConditionalReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = FGSFDQN(
            sf,
            buffer,
            gamma=gamma,
            epsilon=epsilon,
            T=T,
            encoding="task",
            algorithm="alg2",
            n_averaging=1,
        )

    elif method_name == "alg3":
        sf = DeepFGSF(
            input_dim=input_dim,
            n_actions=n_actions,
            n_features=n_features,
            learning_rate=1e-3,
            learning_rate_prior=1e-5,
            learning_rate_w=0.5,
            device=device,
            use_true_reward=False,
            reward_model="linear",
            successor_representation="sf",
            reward_input_dim=n_features,
            sfr_center_threshold=0.05,
        )
        buffer = ConditionalReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        agent = FGSFDQN(
            sf,
            buffer,
            gamma=gamma,
            epsilon=epsilon,
            T=T,
            encoding="task",
            algorithm="alg3",
            n_averaging=5,
        )

    else:
        raise ValueError(f"Unsupported method: {method_name}")

    agent.reset()
    for task in tasks:
        agent.add_training_task(task)

    # Activate the first task before timing.
    agent.set_active_training_task(0)
    return agent


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_timed_steps(method_name: str, agent, tasks, steps: int, device):
    times_ms = []

    # alg2/alg3 need each task initialized once.
    if method_name in ("alg2", "alg3"):
        for i in range(len(tasks)):
            agent.set_active_training_task(i, True)
            agent.active_task.initialize()
        agent.set_active_training_task(0, True)

    for step_idx in range(steps):
        synchronize_if_cuda(device)
        t0 = time.perf_counter()

        if method_name in ("alg2", "alg3"):
            i = np.random.randint(len(tasks))
            agent.set_active_training_task(i, False)
            agent.next_sample()

            if method_name == "alg2":
                batch = agent.buffer.replay()
                if batch:
                    agent._update_batch_grouped_by_prior(batch, i)
            else:  # alg3
                pivot = agent.buffer.sample_pivot()
                if pivot:
                    p_s, p_a, _, _, _ = pivot
                    cond_batch = agent.buffer.sample_conditional(p_s, p_a, agent.n_averaging)
                    if cond_batch:
                        _, _, _, c_next_states, _ = cond_batch
                        c = agent.sf.get_averaged_gpi_policy_index(c_next_states, i)
                        task_c = c if c != i else None
                        agent.sf.update_averaged(p_s, p_a, cond_batch, i, task_c)
        else:
            agent.next_sample()

        synchronize_if_cuda(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)

    arr = np.asarray(times_ms, dtype=float)
    return arr


def summarize_times(method_name, times_ms, warmup=512):
    if len(times_ms) <= warmup:
        return float(np.mean(times_ms)), float(np.var(times_ms))
    trimmed = times_ms[warmup:]
    return float(np.mean(trimmed)), float(np.var(trimmed))


def main():
    parser = argparse.ArgumentParser(description="Per-step training-time benchmark (first 100 steps).")
    parser.add_argument("--steps", type=int, default=1000, help="Number of timed steps per method.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n-tasks", type=int, default=8, help="Number of training tasks.")
    parser.add_argument(
        "--methods",
        type=str,
        default="DQN,SFDQN,FGSFDQN,alg2,alg3,FGDQN",
        help="Comma-separated list from: DQN,SFDQN,FGSFDQN,alg2,alg3,FGDQN",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Replay batch size.")
    parser.add_argument("--buffer-size", type=int, default=200000, help="Replay buffer size.")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = build_gridworld_tasks(args.n_tasks)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print("=" * 68)
    print("Per-step training-time benchmark")
    print(f"device={device} | steps={args.steps} | n_tasks={args.n_tasks}")
    print("=" * 68)

    summary = {}

    for method in methods:
        print("\n" + "-" * 68)
        print(f"Method: {method}")
        print("-" * 68)

        set_global_seed(args.seed)
        agent = init_agent(method, tasks, args.batch_size, args.buffer_size, device)
        times_ms = run_timed_steps(method, agent, tasks, args.steps, device)

        mean_ms, var_ms = summarize_times(method, times_ms, warmup=512)
        summary[method] = (mean_ms, var_ms)

        print(f"{method} summary over first {args.steps} steps:")
        print(f"mean_ms={mean_ms:.6f}")
        print(f"var_ms={var_ms:.6f}")

    print("\n" + "=" * 68)
    print(f"FINAL SUMMARY (first {args.steps} timed steps per method)")
    print("=" * 68)
    for method in methods:
        mean_ms, var_ms = summary[method]
        print(f"{method:10s} | mean_ms={mean_ms:.6f} | var_ms={var_ms:.6f}")


if __name__ == "__main__":
    main()
