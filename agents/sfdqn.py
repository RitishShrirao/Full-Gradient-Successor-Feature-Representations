# -*- coding: UTF-8 -*-
import torch
import random
import numpy as np
import time

from agents.agent import Agent


class SFDQN(Agent):
    
    def __init__(self, deep_sf, buffer, *args, use_gpi=True, test_epsilon=0.03, **kwargs):
        """
        Creates a new SFDQN agent per the specifications in the original paper.
        
        Parameters
        ----------
        deep_sf : DeepSF
            instance of deep successor feature representation
         buffer : ReplayBuffer
            a replay buffer that implements randomized experience replay
        use_gpi : boolean
            whether or not to use transfer learning (defaults to True)
        test_epsilon : float
            the exploration parameter for epsilon greedy used during testing 
            (defaults to 0.03 as in the paper)
        """
        super(SFDQN, self).__init__(*args, **kwargs)
        self.sf = deep_sf
        self.buffer = buffer
        self.use_gpi = use_gpi
        self.test_epsilon = test_epsilon
        
    def get_Q_values(self, s, s_enc):
    # Ensure s_enc is numpy array with batch-dim
        s_enc_arr = np.asarray(s_enc)
        if s_enc_arr.ndim == 1:
            s_enc_arr = s_enc_arr[None, ...]  # [1, ...]
        # Call SF.GPI with numpy (we made SF.GPI expect numpy)
        q, c = self.sf.GPI(s_enc_arr, self.task_index, update_counters=self.use_gpi)
        if not self.use_gpi:
            c = np.asarray(self.task_index)
        self.c = c
        # q: [B, n_tasks, n_actions] as numpy or torch
        if isinstance(q, torch.Tensor):
            q_np = q.detach().cpu().numpy()
        else:
            q_np = np.asarray(q)
        c_arr_all = np.asarray(c)
        if c_arr_all.ndim == 0:
            q_sel = q_np[:, int(c_arr_all.item()), :]   # [B, n_actions]
        else:
            c_arr = c_arr_all.astype(np.int64, copy=False)
            B = q_np.shape[0]
            q_sel = q_np[np.arange(B), c_arr, :]   # [B, n_actions]
        return q_sel

    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        # update w
        phi = self.phi(s, a, s1)
        if isinstance(phi, torch.Tensor):
            phi = phi.detach().cpu().numpy()
        phi_target = self.sf.encode_transition_feature(phi)
        if isinstance(s_enc, torch.Tensor): 
            s_enc = s_enc.detach().cpu().numpy()
        
        if isinstance(s1_enc, torch.Tensor): 
            s1_enc = s1_enc.detach().cpu().numpy()
        self.sf.update_reward(phi, r, self.task_index)
        # remember this experience
        self.buffer.append(s_enc, a, phi_target, s1_enc, gamma)
        # update SFs
        transitions = self.buffer.replay()
        for index in range(self.n_tasks):
            self.sf.update_successor(transitions, index)
        
    def reset(self):
        super(SFDQN, self).reset()
        self.sf.reset()
        self.buffer.reset()

    def add_training_task(self, task):
        super(SFDQN, self).add_training_task(task)
        self.sf.add_training_task(task, source=None)
    
    def get_progress_strings(self):
        sample_str, reward_str = super(SFDQN, self).get_progress_strings()
        gpi_percent = self.sf.GPI_usage_percent(self.task_index)
        if self.sf.reward_model == 'linear':
            fit_w = self.sf.fit_w[self.task_index]
            true_w = self.sf.true_w[self.task_index]
            if isinstance(true_w, torch.Tensor):
                true_w = true_w.detach().cpu().numpy()
            if isinstance(fit_w, torch.Tensor):
                fit_w = fit_w.detach().cpu().numpy()
            w_error = np.linalg.norm(fit_w - true_w)
            gpi_str = 'GPI% \t {:.4f} \t w_err \t {:.4f}'.format(gpi_percent, w_error)
        else:
            r_err = float(self.sf.reward_fit_error[self.task_index])
            gpi_str = 'GPI% \t {:.4f} \t r_fit_err \t {:.6f}'.format(gpi_percent, r_err)
        return sample_str, reward_str, gpi_str
            
    def train(self, train_tasks, n_samples, viewers=None, n_view_ev=None, test_tasks=[], n_test_ev=1000):
        if viewers is None: 
            viewers = [None] * len(train_tasks)
            
        # add tasks
        self.reset()
        for train_task in train_tasks:
            self.add_training_task(train_task)
            
        # train each one
        return_data = []
        for index, (train_task, viewer) in enumerate(zip(train_tasks, viewers)):
            self.set_active_training_task(index)
            for t in range(n_samples):
                
                # train
                self.next_sample(viewer, n_view_ev)
                
                # test
                if t % n_test_ev == 0:
                    Rs = []
                    for test_task in test_tasks:
                        R = self.test_agent(test_task)
                        Rs.append(R)
                    print('test performance: {}'.format('\t'.join(map('{:.4f}'.format, Rs))))
                    avg_R = np.mean(Rs)
                    return_data.append(avg_R)
        return return_data
    
    def get_test_action(self, s_enc, w):
        if random.random() <= self.test_epsilon:
            return random.randrange(self.n_actions)
        # ensure s_enc is numpy batch
        s_enc_arr = np.asarray(s_enc)
        if s_enc_arr.ndim == 1:
            s_enc_arr = s_enc_arr[None, ...]
        if self.sf.reward_model == 'linear':
            q, c = self.sf.GPI_w(s_enc_arr, w, task_index=self.task_index)  # q: [B, n_tasks, n_actions]
        else:
            q, c = self.sf.GPI_w(s_enc_arr, task_index=self.task_index)
        if isinstance(q, torch.Tensor):
            q_np = q.detach().cpu().numpy()
        else:
            q_np = np.asarray(q)
        c_arr_all = np.asarray(c)
        if c_arr_all.ndim == 0:
            q_sel = q_np[:, int(c_arr_all.item()), :]
        else:
            c_arr = c_arr_all.astype(np.int64, copy=False)
            B = q_np.shape[0]
            q_sel = q_np[np.arange(B), c_arr, :]
        # return argmax for first state (test uses single-state)
        a = int(np.argmax(q_sel[0]))
        return a

            
    def test_agent(self, task, return_history=False, visualize=False, pause=0.12, max_steps=None):
        """
        Run one episode with the agent.

        Args:
            task: environment/task object
            return_history (bool): if True, return detailed episode dict
            visualize (bool): if True, render the episode using rich (replay from history)
            pause (float): seconds between frames when visualizing
            max_steps (int|None): override self.T

        Returns:
            If return_history=False: total reward (float).
            If return_history=True: dict with:
                - total_reward (float)
                - steps (int)
                - states (list of states)
                - actions (list of actions)
                - rewards (list of rewards)
        """
        T = max_steps if max_steps is not None else self.T

        w = task.get_w()
        total_reward = 0.0
        s = task.initialize()
        s_enc = self.encoding(s)

        states = [s]     
        actions = []
        rewards = []

        for step in range(T):
            a = self.get_test_action(s_enc, w)
            s1, r, done = task.transition(a)

            actions.append(a)
            rewards.append(r)
            total_reward += r

            states.append(s1)
            s_enc = self.encoding(s1)
            s = s1

            if done:
                break

        episode = {
            "total_reward": total_reward,
            "steps": len(actions),
            "states": states,
            "actions": actions,
            "rewards": rewards
        }

        if visualize:
            Agent.render_episode_history_rich(episode, task, agent=self, pause=pause)

        if return_history:
            return episode
        return total_reward
    
