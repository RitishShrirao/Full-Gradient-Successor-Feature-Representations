# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _TorchRewardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, learning_rate=1e-3):
        super(_TorchRewardMLP, self).__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class SF:
    
    def __init__(self, learning_rate_w, *args, use_true_reward=False,
                 reward_model='linear', reward_hidden_units=128,
                 successor_representation='sf', reward_input_dim=None,
                 sfr_center_threshold=0.0, reward_learning_rate=None, **kwargs):
        """
        Creates a new abstract successor feature representation.
        
        Parameters
        ----------
        learning_rate_w : float
            the learning rate to use for learning the reward weights using gradient descent
        use_true_reward : boolean
            whether or not to use the true reward weights from the environment, or learn them
            using gradient descent
        """
        self.alpha_w = learning_rate_w
        self.use_true_reward = use_true_reward
        self.reward_model = str(reward_model).strip().lower()
        if self.reward_model not in ['linear', 'nonlinear']:
            raise ValueError('reward_model must be one of: linear, nonlinear')
        self.reward_hidden_units = int(reward_hidden_units)
        self.successor_representation = str(successor_representation).strip().lower()
        if self.successor_representation not in ['sf', 'sfr']:
            raise ValueError('successor_representation must be one of: sf, sfr')
        self.reward_input_dim = reward_input_dim
        self.sfr_center_threshold = float(sfr_center_threshold)
        if reward_learning_rate is None:
            self.reward_learning_rate = 1e-3
        else:
            self.reward_learning_rate = float(reward_learning_rate)
        if len(args) != 0 or len(kwargs) != 0:
            print(self.__class__.__name__ + ' ignoring parameters ' + str(args) + ' and ' + str(kwargs))
        self.reset()
            
    def build_successor(self, task, source=None):
        """
        Builds a new successor feature map for the specified task. This method should not be called directly.
        Instead, add_task should be called instead.
        
        Parameters
        ----------
        task : Task
            a new MDP environment for which to learn successor features
        source : integer
            if specified and not None, the parameters of the successor features for the task at the source
            index should be copied to the new successor features, as suggested in [1]
        
        Returns
        -------
        object : the successor feature representation for the new task, which can be a torch model, 
        a lookup table (dictionary) or another learning representation
        """
        raise NotImplementedError
        
    def get_successor(self, state, policy_index):
        """
        Evaluates the successor features in given states for the specified task.
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        policy_index : integer
            the index of the task whose successor features to evaluate
        
        Returns
        -------
        np.ndarray : the evaluation of the successor features, which is of shape
        [n_batch, n_actions, n_features], where
            n_batch is the number of states in the state argument
            n_actions is the number of actions of the MDP
            n_features is the number of features in the SF representation
        """
        raise NotImplementedError
    
    def get_successors(self, state):
        """
        Evaluates the successor features in given states for all tasks.
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        
        Returns
        -------
        np.ndarray : the evaluation of the successor features, which is of shape
        [n_batch, n_tasks, n_actions, n_features], where
            n_batch is the number of states in the state argument
            n_tasks is the number of tasks
            n_actions is the number of actions of the MDP
            n_features is the number of features in the SF representation
        """
        raise NotImplementedError
    
    def update_successor(self, transitions, policy_index):
        """
        Updates the successor representation by training it on the given transition.
        
        Parameters
        ----------
        transitions : object
            collection of transitions
        policy_index : integer
            the index of the task whose successor features to update
        """
        raise NotImplementedError
        
    def reset(self):
        """
        Removes all trained successor feature representations from the current object, all learned rewards,
        and all task information.
        """
        self.n_tasks = 0
        self.n_features = getattr(self, 'n_features', 0)
        self.psi = []
        self.true_w = []
        self.fit_w = []
        self.reward_models = []
        self.reward_optimizers = []
        self.reward_fit_error = []
        self.reward_support = []
        self.sfr_centers = []
        self.sfr_center_reward_sums = []
        self.sfr_center_reward_counts = []
        self.gpi_counters = []

    def _get_device(self):
        dev = getattr(self, 'device', None)
        if dev is None:
            return torch.device('cpu')
        if isinstance(dev, torch.device):
            return dev
        return torch.device(dev)

    def add_training_task(self, task, source=None):
        """
        Adds a successor feature representation for the specified task.
        
        Parameters
        ----------
        task : Task
            a new MDP environment for which to learn successor features
        source : integer
            if specified and not None, the parameters of the successor features for the task at the source
            index should be copied to the new successor features, as suggested in [1]
        """
        
        # add successor features to the library
        self.psi.append(self.build_successor(task, source))
        self.n_tasks = len(self.psi)
        
        # build new reward function
        true_w = task.get_w()
        self.true_w.append(true_w)
        reward_dim = int(self.reward_input_dim) if self.reward_input_dim is not None else int(task.feature_dim())
        if self.reward_model == 'linear':
            if self.use_true_reward:
                fit_w = np.asarray(true_w).reshape(-1, 1)
            else:
                fit_w = np.random.uniform(low=-0.01, high=0.01, size=(reward_dim, 1))
            self.reward_models.append(None)
            self.reward_optimizers.append(None)
        else:
            fit_w = None
            model = _TorchRewardMLP(
                input_dim=reward_dim,
                hidden_dim=self.reward_hidden_units,
                learning_rate=self.reward_learning_rate,
            ).to(self._get_device())
            self.reward_models.append(model)
            self.reward_optimizers.append(optim.Adam(model.parameters(), lr=self.reward_learning_rate))
        self.fit_w.append(fit_w)
        self.reward_fit_error.append(0.0)
        self.reward_support.append(np.zeros((int(self.n_features),), dtype=float))
        self._refresh_reward_support(task_index=self.n_tasks - 1)
        
        # add statistics
        for i in range(len(self.gpi_counters)):
            self.gpi_counters[i] = np.append(self.gpi_counters[i], 0)
        self.gpi_counters.append(np.zeros((self.n_tasks,), dtype=int))
        
    def update_reward(self, phi, r, task_index, exact=False):
        phi_arr = np.asarray(phi, dtype=float).reshape(-1)
        if self.reward_model == 'linear':
            w = np.asarray(self.fit_w[task_index])
            # Ensure column vector shape (n_features, 1)
            w = w.reshape(-1, 1)
            phi_col = phi_arr.reshape(w.shape)
            r_fit = float(np.sum(phi_col * w))
            self.fit_w[task_index] = (w + self.alpha_w * (r - r_fit) * phi_col).reshape(w.shape)
            self.reward_fit_error[task_index] = float((r - r_fit) ** 2)
            # validate reward
            r_true = float(np.sum(phi_col * np.asarray(self.true_w[task_index]).reshape(w.shape)))
            if exact and not np.allclose(r, r_true):
                raise Exception('sampled reward {} != linear reward {} - please check task {}!'.format(
                    r, r_true, task_index))
            self._refresh_reward_support(task_index)
            return

        target = np.asarray([[float(r)]], dtype=float)
        model = self.reward_models[task_index]
        optimizer = self.reward_optimizers[task_index]
        model.train()
        x_t = torch.as_tensor(phi_arr.reshape(1, -1), dtype=torch.float32, device=self._get_device())
        y_t = torch.as_tensor(target, dtype=torch.float32, device=self._get_device())
        pred = model(x_t)
        loss = nn.MSELoss()(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        self.reward_fit_error[task_index] = float(loss.item())
        self._sfr_update_reward_stats(phi_arr, r)
        self._refresh_reward_support(task_index)

    def _reward_from_phi(self, phi, task_index):
        phi_arr = np.asarray(phi, dtype=float).reshape(1, -1)
        if self.reward_model == 'linear':
            w = np.asarray(self.fit_w[task_index]).reshape(-1, 1)
            return float(np.sum(phi_arr.reshape(-1, 1) * w))
        model = self.reward_models[task_index]
        model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(phi_arr, dtype=torch.float32, device=self._get_device())
            return float(model(x_t).reshape(-1)[0].item())

    def _refresh_reward_support(self, task_index):
        if self.successor_representation != 'sfr':
            return
        support = np.zeros((int(self.n_features),), dtype=float)
        for idx, center in enumerate(self.sfr_centers[: int(self.n_features)]):
            if idx < len(self.sfr_center_reward_counts) and self.sfr_center_reward_counts[idx] > 0:
                support[idx] = self.sfr_center_reward_sums[idx] / max(1, self.sfr_center_reward_counts[idx])
            else:
                support[idx] = self._reward_from_phi(center, task_index)
        self.reward_support[task_index] = support

    def _sfr_nearest_center_idx(self, phi):
        if len(self.sfr_centers) == 0:
            return None
        phi_arr = np.asarray(phi, dtype=float).reshape(-1)
        centers = np.asarray(self.sfr_centers)
        dists = np.linalg.norm(centers - phi_arr, axis=1)
        return int(np.argmin(dists))

    def _sfr_update_reward_stats(self, phi, r):
        if self.successor_representation != 'sfr':
            return
        idx = self._sfr_nearest_center_idx(phi)
        if idx is None:
            return
        while len(self.sfr_center_reward_sums) <= idx:
            self.sfr_center_reward_sums.append(0.0)
            self.sfr_center_reward_counts.append(0)
        self.sfr_center_reward_sums[idx] += float(r)
        self.sfr_center_reward_counts[idx] += 1

    def _sfr_assign_center(self, phi):
        phi_arr = np.asarray(phi, dtype=float).reshape(-1)
        max_centers = int(self.n_features)
        if len(self.sfr_centers) == 0:
            self.sfr_centers.append(phi_arr.copy())
            self.sfr_center_reward_sums.append(0.0)
            self.sfr_center_reward_counts.append(0)
            return 0, True

        centers = np.asarray(self.sfr_centers)
        dists = np.linalg.norm(centers - phi_arr, axis=1)
        idx = int(np.argmin(dists))
        added = False
        if len(self.sfr_centers) < max_centers and float(dists[idx]) > self.sfr_center_threshold:
            self.sfr_centers.append(phi_arr.copy())
            self.sfr_center_reward_sums.append(0.0)
            self.sfr_center_reward_counts.append(0)
            idx = len(self.sfr_centers) - 1
            added = True
        return idx, added

    def encode_transition_feature(self, phi):
        """
        Returns the transition feature used in Bellman targets.

        - SF mode: raw feature vector phi
        - SFR mode: one-hot bin over learned successor-feature prototypes
        """
        phi_arr = np.asarray(phi, dtype=float).reshape(-1)
        if self.successor_representation != 'sfr':
            return phi_arr

        idx, added = self._sfr_assign_center(phi_arr)
        if added:
            for task_i in range(self.n_tasks):
                self._refresh_reward_support(task_i)
        one_hot = np.zeros((int(self.n_features),), dtype=float)
        one_hot[idx] = 1.0
        return one_hot

    def score_successor(self, successor_values, task_index, w=None):
        """
        Scores successor descriptors to produce Q-values.

        Linear mode: Q = <psi, w>
        Nonlinear mode: Q ~= R(xi), where xi is the learned successor descriptor.
        """
        if isinstance(successor_values, torch.Tensor):
            sv_t = successor_values
            if self.successor_representation == 'sfr':
                support = np.asarray(self.reward_support[task_index]).reshape(-1)
                support_t = torch.as_tensor(support, dtype=sv_t.dtype, device=sv_t.device)
                if support_t.shape[0] != sv_t.shape[-1]:
                    if support_t.shape[0] < sv_t.shape[-1]:
                        support_t = torch.nn.functional.pad(support_t, (0, sv_t.shape[-1] - support_t.shape[0]))
                    else:
                        support_t = support_t[:sv_t.shape[-1]]
                return torch.tensordot(sv_t, support_t, dims=([-1], [0]))

            if self.reward_model == 'linear':
                if w is None:
                    w = self.fit_w[task_index]
                w_t = torch.as_tensor(np.asarray(w).reshape(-1), dtype=sv_t.dtype, device=sv_t.device)
                return torch.tensordot(sv_t, w_t, dims=([-1], [0]))

            model = self.reward_models[task_index]
            model.eval()
            with torch.no_grad():
                f_dim = sv_t.shape[-1]
                flat = sv_t.reshape(-1, f_dim)
                q_flat = model(flat).reshape(-1)
                return q_flat.reshape(sv_t.shape[:-1])

        sv = np.asarray(successor_values)
        if self.successor_representation == 'sfr':
            support = np.asarray(self.reward_support[task_index]).reshape(-1)
            if support.shape[0] != sv.shape[-1]:
                if support.shape[0] < sv.shape[-1]:
                    support = np.pad(support, (0, sv.shape[-1] - support.shape[0]))
                else:
                    support = support[:sv.shape[-1]]
            return np.tensordot(sv, support, axes=([-1], [0]))

        if self.reward_model == 'linear':
            if w is None:
                w = self.fit_w[task_index]
            w_arr = np.asarray(w).reshape(-1)
            return np.tensordot(sv, w_arr, axes=([-1], [0]))

        model = self.reward_models[task_index]
        model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(sv, dtype=torch.float32, device=self._get_device())
            f_dim = x_t.shape[-1]
            flat = x_t.reshape(-1, f_dim)
            q_flat = model(flat).reshape(-1)
            q = q_flat.reshape(x_t.shape[:-1])
        return q.detach().cpu().numpy()

    def get_xi(self, state, policy_index):
        # Alias for SFR-style notation.
        return self.get_successor(state, policy_index)

    def get_xis(self, state):
        # Alias for SFR-style notation.
        return self.get_successors(state)

    
    def GPE_w(self, state, policy_index, w=None, task_index=None):
        """
        Implements generalized policy evaluation according to [1]. In summary, this uses the
        learned reward parameters of one task and successor features of a policy to estimate the Q-values of 
        the policy if it were executed in that task.
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        policy_index : integer
            the index of the task whose policy to evaluate
        w : numpy array
            reward parameters of the task in which to evaluate the policy
            
        Returns
        -------
        np.ndarray : the estimated Q-values of shape [n_batch, n_actions], where
            n_batch is the number of states in the state argument
            n_actions is the number of actions in the MDP            
        """
        psi = self.get_successor(state, policy_index)  # expected np.ndarray [B, A, F]
        if task_index is None:
            task_index = policy_index
        q = self.score_successor(psi, task_index=task_index, w=w)
        return q
        
    def GPE(self, state, policy_index, task_index):
        """
        Implements generalized policy evaluation according to [1]. In summary, this uses the
        learned reward parameters of one task and successor features of a policy to estimate the Q-values of 
        the policy if it were executed in that task.
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        policy_index : integer
            the index of the task whose policy to evaluate
        task_index : integer
            the index of the task (e.g. reward) to use to evaluate the policy
            
        Returns
        -------
        np.ndarray : the estimated Q-values of shpae [n_batch, n_actions], where
            n_batch is the number of states in the state argument
            n_actions is the number of actions in the MDP            
        """
        if self.reward_model == 'linear':
            return self.GPE_w(state, policy_index, self.fit_w[task_index], task_index=task_index)
        return self.GPE_w(state, policy_index, task_index=task_index)
    
    def GPI_w(self, state, w=None, task_index=None):
        """
        Implements generalized policy improvement according to [1]. 
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        w : numpy array
            the reward parameters of the task to control
        
        Returns
        -------
        np.ndarray : the maximum Q-values computed by GPI for selecting actions
        of shape [n_batch, n_tasks, n_actions], where:
            n_batch is the number of states in the state argument
            n_tasks is the number of tasks
            n_actions is the number of actions in the MDP 
        np.ndarray : the tasks that are active in each state of state_batch in GPi
        """
        if task_index is None:
            if self.reward_model == 'nonlinear':
                raise ValueError('task_index is required for nonlinear reward_model in GPI_w')
            task_index = 0

        psi = self.get_successors(state)
        q = self.score_successor(psi, task_index=task_index, w=w)  # [B, n_tasks, n_actions]
        # choose best task per state: max over actions then argmax over tasks
        if isinstance(q, torch.Tensor):
            task = torch.argmax(torch.max(q, dim=2).values, dim=1)
        else:
            task = np.argmax(np.max(q, axis=2), axis=1)  # shape [B]
        return q, task

    
    def GPI(self, state, task_index, update_counters=False):
        """
        Implements generalized policy improvement according to [1]. 
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        task_index : integer
            the index of the task in which the GPI action will be used
        update_counters : boolean
            whether or not to keep track of which policies are active in GPI
        
        Returns
        -------
        np.ndarray : the maximum Q-values computed by GPI for selecting actions
        of shape [n_batch, n_tasks, n_actions], where:
            n_batch is the number of states in the state argument
            n_tasks is the number of tasks
            n_actions is the number of actions in the MDP 
        np.ndarray : the tasks that are active in each state of state_batch in GPi
        """
        if self.reward_model == 'linear':
            q, task = self.GPI_w(state, self.fit_w[task_index], task_index=task_index)
        else:
            q, task = self.GPI_w(state, task_index=task_index)
        if update_counters:
            if isinstance(task, torch.Tensor):
                task_idx = task.detach().cpu().numpy().astype(np.int64)
            else:
                task_idx = np.asarray(task).astype(np.int64)
            self.gpi_counters[task_index][task_idx] += 1
        return q, task
    
    def GPI_usage_percent(self, task_index):
        """
        Counts the number of times that actions were transferred from other tasks.
        
        Parameters
        ----------
        task_index : integer
            the index of the task
        
        Returns
        -------
        float : the (normalized) number of actions that were transferred from other
            tasks in GPi.
        """
        counts = self.gpi_counters[task_index]
        denom = np.sum(counts)
        if denom == 0:
            return 0.0
        return 1. - (float(counts[task_index]) / denom)