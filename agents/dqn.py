import numpy as np
import random

import torch

from agents.agent import Agent
import time

class DQN(Agent):
    
    def __init__(self, model_lambda, buffer, *args, target_update_ev=1000, test_epsilon=0.03, **kwargs):
        """
        Creates a new DQN agent that supports universal value function approximation (UVFA).
        
        Parameters
        ----------
        model_lambda : function
            returns a torch model
        buffer : ReplayBuffer
            a replay buffer that implements randomized experience replay
        target_update_ev : integer
            how often to update the target network (defaults to 1000)  (alpha / beta)
        test_epsilon : float
            the exploration parameter for epsilon greedy used during testing 
            (defaults to 0.03 as in the paper)
        """
        super(DQN, self).__init__(*args, **kwargs)
        self.model_lambda = model_lambda
        self.buffer = buffer
        self.target_update_ev = target_update_ev
        self.test_epsilon = test_epsilon
    
    def reset(self):
        Agent.reset(self)
        self.Q = self.model_lambda()
        self.target_Q = self.model_lambda()
        # self.target_Q.set_weights(self.Q.get_weights())
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.buffer.reset()
        self.updates_since_target_updated = 0
        
    def get_Q_values(self, s, s_enc):
        # return self.Q.predict_on_batch(s_enc)
        with torch.no_grad():
            s_enc_tensor = torch.from_numpy(s_enc).float()
            q_values = self.Q(s_enc_tensor)
        return q_values.cpu().numpy()

    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        # remember this experience
        self.buffer.append(s_enc, a, r, s1_enc, gamma)

        # sample experience at random
        batch = self.buffer.replay()
        if batch is None:
            return
        states, actions, rewards, next_states, gammas = batch

        # Convert to tensors
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float().flatten()
        next_states = torch.from_numpy(next_states).float()
        gammas = torch.from_numpy(gammas).float()

        # Move all tensors to the same device as the model
        device = self.Q.network[0].weight.device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        gammas = gammas.to(device)

        n_batch = self.buffer.n_batch
        indices = np.arange(n_batch)
        rewards = rewards.flatten()

        # main update
        self.Q.optimizer.zero_grad()
        q_values = self.Q(states)
        q_values_next = self.Q(next_states)
        target_q_values_next = self.target_Q(next_states)
        next_actions = torch.argmax(q_values_next, dim=1)

        q_value_targets = q_values.clone().detach()
        q_value_targets[indices, actions] = rewards + gammas * target_q_values_next[indices, next_actions]

        loss = self.Q.loss(q_values, q_value_targets)
        loss.backward()
        self.Q.optimizer.step()

        # target update
        self.updates_since_target_updated += 1
        if self.updates_since_target_updated >= self.target_update_ev:
            self.target_Q.load_state_dict(self.Q.state_dict())
            self.updates_since_target_updated = 0
    
    def train(self, train_tasks, n_samples, viewers=None, n_view_ev=None, test_tasks=[], n_test_ev=1000):
        if viewers is None: 
            viewers = [None] * len(train_tasks)
        
        
        # add tasks
        self.reset()
        for train_task in train_tasks:
            self.add_training_task(train_task)
        print(train_task)    
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
                    avg_R = np.mean(Rs)
                    return_data.append(avg_R)
                    print('test performance: {}'.format('\t'.join(map('{:.4f}'.format, Rs))))
        return return_data
    
    def get_test_action(self, s_enc):
        if random.random() <= self.test_epsilon:
            a = random.randrange(self.n_actions)
        else:
            q = self.get_Q_values(s_enc, s_enc)
            a = np.argmax(q)
        return a
            
    def test_agent(self, task):
        R = 0.
        s = task.initialize()
        s_enc = self.encoding(s)
        for _ in range(self.T):
            a = self.get_test_action(s_enc)
            s1, r, done = task.transition(a)
            s1_enc = self.encoding(s1)
            s, s_enc = s1, s1_enc
            R += r
            if done:
                break
        return R


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

        total_reward = 0.0
        s = task.initialize()
        s_enc = self.encoding(s)

        states = [s]     
        actions = []
        rewards = []

        for step in range(T):
            a = self.get_test_action(s_enc)
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
