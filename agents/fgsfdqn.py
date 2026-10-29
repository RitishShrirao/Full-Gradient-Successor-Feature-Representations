import numpy as np
import torch
from agents.sfdqn import SFDQN

class FGSFDQN(SFDQN):
    def __init__(self, deep_sf, buffer, *args, 
                 algorithm='alg1',
                 n_averaging=1,
                 pivot_batch_size=None,
                 **kwargs):
        """
        algorithm: 'alg1' (sequential), 'alg2' (randomized tasks), 'alg3' (randomized + averaging),
                   'alg4' / 'alg1_averaged' (sequential + averaging).
        n_averaging: number of next-state samples averaged per pivot (1 = single sample).
        pivot_batch_size: number of pivots per averaged update (defaults to the buffer batch size).
        """
        super(FGSFDQN, self).__init__(deep_sf, buffer, *args, **kwargs)
        self.algorithm = algorithm
        self.n_averaging = n_averaging
        self.pivot_batch_size = pivot_batch_size

    def _full_gradient_update(self, task_i):
        """One full-gradient update of the current task and its GPI-selected priors."""
        if self.n_averaging > 1 and hasattr(self.buffer, "sample_batch_conditional"):
            n_pivots = self.pivot_batch_size or self.buffer.n_batch
            batch_data = self.buffer.sample_batch_conditional(n_pivots, self.n_averaging)
            if batch_data is not None:
                self._maybe_diagnose(batch_data, task_i)
                self.sf.update_averaged_batch(batch_data, task_i)
            return
        batch = self.buffer.replay()
        if batch is not None:
            self._maybe_diagnose(batch, task_i)
            self.sf.update_full_gradient(batch, task_i)

    def _update_batch_grouped_by_prior(self, batch, task_i):
        # Kept for backward compatibility (benchmark_step_time.py).
        self.sf.update_full_gradient(batch, task_i)

    def _append_resamples(self, task, s_enc, a):
        """Stores the extra next states that the task sampled from the same (s, a) (simulator resampling)."""
        for s1x, phix, terminal in getattr(task, "last_resamples", None) or ():
            phi_target = self.sf.encode_transition_feature(np.asarray(phix, dtype=np.float32))
            self.buffer.append(s_enc, a, phi_target, self.encoding(s1x), 0.0 if terminal else self.gamma)

    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        # Store transition + reward update for all algorithms
        phi = self.phi(s, a, s1)
        if isinstance(phi, torch.Tensor):
            phi = phi.detach().cpu().numpy()
        phi_target = self.sf.encode_transition_feature(phi)
        self.sf.update_reward(phi, r, self.task_index)
        self.buffer.append(s_enc, a, phi_target, s1_enc, gamma)
        self._append_resamples(self.active_task, s_enc, a)

        # Sequential algorithms (Alg 1, and Alg 1 + averaging) update here;
        # Alg 2/3 update inside train_randomized.
        if self.algorithm in ('alg1', 'alg4', 'alg1_averaged'):
            self._full_gradient_update(self.task_index)

    def train_randomized(self, train_tasks, n_total_steps, viewers=None, n_view_ev=None):
        if viewers is None: viewers = [None] * len(train_tasks)
        self.reset()
        # all tasks are added up front here, so copying would only duplicate the initial network
        copy_flag, self.copy_previous_sf = getattr(self, 'copy_previous_sf', False), False
        for t in train_tasks: self.add_training_task(t)
        self.copy_previous_sf = copy_flag
            
        task_states = [task.initialize() for task in train_tasks]
        task_states_enc = [self.encoding(s) for s in task_states]
        
        self.set_active_training_task(0, reset=True)
        
        # Warmup buffer with valid transitions across all tasks
        warmup_per_task = max(1, self.buffer.n_batch // len(train_tasks))
        for i in range(len(train_tasks)):
            self.set_active_training_task(i, reset=False)
            s = task_states[i]
            s_enc = task_states_enc[i]
            for _ in range(warmup_per_task):
                a = np.random.randint(self.n_actions)
                s1, r, terminal = train_tasks[i].transition(a)
                s1_enc = self.encoding(s1)
                gamma = 0.0 if terminal else self.gamma
                
                phi = self.phi(s, a, s1)
                if isinstance(phi, torch.Tensor):
                    phi = phi.detach().cpu().numpy()
                phi_target = self.sf.encode_transition_feature(phi)
                self.sf.update_reward(phi, r, i)
                self.buffer.append(s_enc, a, phi_target, s1_enc, gamma)
                self._append_resamples(train_tasks[i], s_enc, a)
                
                if terminal:
                    s = train_tasks[i].initialize()
                    s_enc = self.encoding(s)
                else:
                    s = s1
                    s_enc = s1_enc
            task_states[i] = s
            task_states_enc[i] = s_enc

        self.total_tasks = len(train_tasks)
        self.n_samples = n_total_steps
        self.reward_since_last_episode = 0.0
        episode_steps = [0] * len(train_tasks)  # per-task steps since the last reset (horizon T)

        for t in range(n_total_steps):
            i = np.random.randint(len(train_tasks))
            self.set_active_training_task(i, reset=False)
            
            s = task_states[i]
            s_enc = task_states_enc[i]
            
            q = self.get_Q_values(s, s_enc)
            a = self._epsilon_greedy(q)
            
            s1, r, terminal = train_tasks[i].transition(a)
            s1_enc = self.encoding(s1)
            gamma = 0.0 if terminal else self.gamma
            
            phi = self.phi(s, a, s1)
            if isinstance(phi, torch.Tensor):
                phi = phi.detach().cpu().numpy()
            phi_target = self.sf.encode_transition_feature(phi)
            self.sf.update_reward(phi, r, i)
            self.buffer.append(s_enc, a, phi_target, s1_enc, gamma)
            self._append_resamples(train_tasks[i], s_enc, a)
            
            self.steps = t + 1
            self.reward += r
            self.cum_reward += r
            self.reward_since_last_episode += r

            if (t + 1) % self.save_ev == 0:
                self.cum_reward_hist.append(self.cum_reward)
                if self.reward_log_callback is not None:
                    try:
                        self.reward_log_callback(
                            step=self.steps,
                            cumulative_reward=self.cum_reward,
                            task_index=-1,
                        )
                    except Exception:
                        pass
            
            episode_steps[i] += 1
            if terminal or episode_steps[i] >= self.T:
                # timeouts keep gamma > 0 (the transition is not terminal), as in Agent.next_sample
                episode_steps[i] = 0
                self.episode += 1
                self.episode_reward = self.reward_since_last_episode
                self.reward_since_last_episode = 0.0
                s_new = train_tasks[i].initialize()
                task_states[i] = s_new
                task_states_enc[i] = self.encoding(s_new)
            else:
                task_states[i] = s1
                task_states_enc[i] = s1_enc

            if (t + 1) % self.print_ev == 0:
                prev_task_idx = self.task_index
                self.task_index = "All"
                msg = self.format_progress_line()
                self.task_index = prev_task_idx
                if self.progress_logger is not None:
                    self.progress_logger(msg)
                else:
                    print(msg, flush=True)
            
            if self.algorithm in ('alg2', 'alg3'):
                # Alg 2: single sample; Alg 3: averaged over n_averaging next states.
                self._full_gradient_update(i)
