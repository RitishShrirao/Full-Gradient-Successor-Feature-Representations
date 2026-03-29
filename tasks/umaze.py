import numpy as np

from tasks.task import Task


class Umaze(Task):
    """PointMaze wrapper with optional discrete action compatibility."""

    def __init__(
        self,
        goal_cells,
        task_index,
        env_id="PointMaze_Large-v3",
        reset_cell=None,
        include_goal_in_state=True,
        use_dense_reward=True,
        allow_discrete_actions=True,
        discrete_action_values=(-1.0, 0.0, 1.0),
        goal_done_threshold=0.45,
        goal_jitter=0.0,
    ):
        self.goal_cells = [tuple(map(int, c)) for c in goal_cells]
        self.task_index = int(task_index)
        self.env_id = env_id
        self.reset_cell = tuple(map(int, reset_cell)) if reset_cell is not None else None
        self.include_goal_in_state = bool(include_goal_in_state)
        self.use_dense_reward = bool(use_dense_reward)
        self.allow_discrete_actions = bool(allow_discrete_actions)
        self.goal_done_threshold = float(goal_done_threshold)
        # Small XY jitter lets nearby tasks share one cell.
        self.goal_jitter = float(goal_jitter)

        self.env = self._make_env()
        self.last_obs_dict = None
        self.goal_xys = self._build_goal_xys()

        if self.allow_discrete_actions:
            self.action_dict = {}
            for ax in discrete_action_values:
                for ay in discrete_action_values:
                    self.action_dict[len(self.action_dict)] = np.array([ax, ay], dtype=np.float32)
        else:
            self.action_dict = None

    def _make_env(self):
        try:
            import gymnasium as gym
            import gymnasium_robotics

            gym.register_envs(gymnasium_robotics)
            return gym.make(self.env_id)
        except Exception as e:
            raise ImportError(
                "Failed to create UMAZE environment. Install dependencies with: "
                "pip install gymnasium gymnasium-robotics mujoco"
            ) from e

    def _unwrap_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env

    def _build_goal_xys(self):
        """Convert goal cells to XY coordinates, with optional jitter."""
        try:
            maze = getattr(self._unwrap_env(), "maze", None)
            if maze is None:
                base = [self._cell_to_xy_fallback(c) for c in self.goal_cells]
            else:
                base = []
                for cell in self.goal_cells:
                    rc = np.array([int(cell[0]), int(cell[1])], dtype=np.int64)
                    xy = maze.cell_rowcol_to_xy(rc)
                    base.append(np.asarray(xy, dtype=np.float32))
            if self.goal_jitter > 0.0:
                # Fixed RNG keeps offsets reproducible per task.
                rng = np.random.RandomState(1234 + self.task_index)
                jittered = []
                for xy in base:
                    perturb = rng.uniform(-self.goal_jitter, self.goal_jitter, size=2)
                    jittered.append(xy + perturb.astype(np.float32))
                return jittered
            return base
        except Exception:
            base = [self._cell_to_xy_fallback(c) for c in self.goal_cells]
            if self.goal_jitter > 0.0:
                rng = np.random.RandomState(1234 + self.task_index)
                return [xy + rng.uniform(-self.goal_jitter, self.goal_jitter, size=2)
                        for xy in base]
            return base

    def clone(self):
        return Umaze(
            goal_cells=self.goal_cells,
            task_index=self.task_index,
            env_id=self.env_id,
            reset_cell=self.reset_cell,
            include_goal_in_state=self.include_goal_in_state,
            use_dense_reward=self.use_dense_reward,
            allow_discrete_actions=self.allow_discrete_actions,
            goal_done_threshold=self.goal_done_threshold,
        )

    def _tasktype(self):
        return 1

    def initialize(self):
        options = {"goal_cell": np.array(self.goal_cells[self.task_index], dtype=np.int64)}
        if self.reset_cell is not None:
            options["reset_cell"] = np.array(self.reset_cell, dtype=np.int64)

        obs_dict, _ = self.env.reset(options=options)
        self.last_obs_dict = obs_dict
        return self._state_from_obs(obs_dict)

    def action_count(self):
        if self.action_dict is None:
            raise ValueError(
                "Continuous-only UMAZE has no finite action_count(). "
                "Set allow_discrete_actions=True if you want DQN-compatible discrete mapping."
            )
        return len(self.action_dict)

    def transition(self, action):
        if self.last_obs_dict is None:
            self.initialize()

        cont_action = self._to_continuous_action(action)
        obs_dict, env_reward, terminated, truncated, _ = self.env.step(cont_action)
        self.last_obs_dict = obs_dict

        next_state = self._state_from_obs(obs_dict)

        if self.use_dense_reward:
            achieved = np.asarray(obs_dict["achieved_goal"], dtype=np.float32)
            desired = np.asarray(obs_dict["desired_goal"], dtype=np.float32)
            dist = np.linalg.norm(achieved - desired)
            reward = float(np.exp(-dist))
            done = bool(dist <= self.goal_done_threshold)
        else:
            reward = float(env_reward)
            done = bool(terminated or truncated)

        return next_state, reward, done

    def encode(self, state):
        return np.asarray(state, dtype=np.float32).reshape(1, -1)

    def encode_dim(self):
        if self.include_goal_in_state:
            return 6  # observation(4) + desired_goal(2)
        return 4

    def features(self, state, action, next_state):
        ns = np.asarray(next_state, dtype=np.float32)
        achieved = ns[:2]
        phi = np.zeros((len(self.goal_cells),), dtype=np.float32)

        for idx, goal_xy in enumerate(self.goal_xys):
            dist = np.linalg.norm(achieved - goal_xy)
            phi[idx] = float(np.exp(-dist))

        return phi

    def feature_dim(self):
        return len(self.goal_cells)

    def get_w(self):
        w = np.zeros((len(self.goal_cells), 1), dtype=np.float32)
        w[self.task_index, 0] = 1.0
        return w

    def _state_from_obs(self, obs_dict):
        observation = np.asarray(obs_dict["observation"], dtype=np.float32)
        if not self.include_goal_in_state:
            return observation
        desired = np.asarray(obs_dict["desired_goal"], dtype=np.float32)
        return np.concatenate([observation, desired]).astype(np.float32)

    def _to_continuous_action(self, action):
        if np.isscalar(action):
            if self.action_dict is None:
                raise ValueError("Received discrete action but allow_discrete_actions=False")
            action_index = int(np.asarray(action).item())
            if action_index not in self.action_dict:
                raise ValueError(f"bad action {action_index}")
            arr = self.action_dict[action_index]
        else:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if arr.shape[0] != 2:
                raise ValueError(f"Expected continuous action shape (2,), got {arr.shape}")

        arr = np.clip(arr, -1.0, 1.0)
        return arr.astype(np.float32)

    def _cell_to_xy(self, cell):
        try:
            maze = getattr(self._unwrap_env(), "maze", None)
            if maze is not None:
                rc = np.array([int(cell[0]), int(cell[1])], dtype=np.int64)
                xy = maze.cell_rowcol_to_xy(rc)
                return np.asarray(xy, dtype=np.float32)
        except Exception:
            pass
        return self._cell_to_xy_fallback(cell)

    def _cell_to_xy_fallback(self, cell):
        row, col = int(cell[0]), int(cell[1])
        return np.array([float(col), float(row)], dtype=np.float32)
