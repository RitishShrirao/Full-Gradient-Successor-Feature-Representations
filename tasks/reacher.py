# -*- coding: UTF-8 -*-
import numpy as np
from pybulletgym.envs.roboschool.robots.robot_bases import MJCFBasedRobot
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene

from tasks.task import Task

class Reacher(Task):
    
    def __init__(
        self,
        target_positions,
        task_index,
        include_target_in_state=False,
        device='cpu',
        allow_discrete_actions=True,
        discrete_action_values=(-1.0, 0.0, 1.0),
    ):
        self.target_positions = target_positions
        self.task_index = task_index
        self.target_pos = target_positions[task_index]
        self.include_target_in_state = include_target_in_state
        self.device = device  # Store device for tensor conversion
        self.env = ReacherBulletEnv(self.target_pos)
        self.allow_discrete_actions = bool(allow_discrete_actions)
        
        if self.allow_discrete_actions:
            self.action_dict = dict()
            for a1 in discrete_action_values:
                for a2 in discrete_action_values:
                    self.action_dict[len(self.action_dict)] = np.array((a1, a2), dtype=np.float32)
        else:
            self.action_dict = None
    
    def _tasktype(self):
        return 1
    
    def clone(self):
        return Reacher(
            self.target_positions,
            self.task_index,
            self.include_target_in_state,
            self.device,
            self.allow_discrete_actions,
        )
    
    def initialize(self):
        state = self.env.reset()
        if self.include_target_in_state:
            return np.concatenate([state.flatten(), self.target_pos])
        else:
            return state
    
    def action_count(self):
        if self.action_dict is None:
            raise ValueError(
                "Continuous-only Reacher has no finite action_count(). "
                "Set allow_discrete_actions=True if you want DQN-compatible discrete mapping."
            )
        return len(self.action_dict)
    
    def transition(self, action):
        real_action = self._to_continuous_action(action)
        new_state, reward, done, _ = self.env.step(real_action)
        
        if self.include_target_in_state:
            return_state = np.concatenate([new_state, self.target_pos])
        else:
            return_state = new_state
            
        return return_state, reward, done
    
    def encode(self, state):
        state_np = np.asarray(state, dtype=np.float32)
        return state_np.reshape(1, -1)
    
    def encode_dim(self):
        if self.include_target_in_state:
            return 6
        else:
            return 4
    
    def features(self, state, action, next_state):
        phi = np.zeros((len(self.target_positions),), dtype=np.float32)
        
        for index, target in enumerate(self.target_positions):
            delta = np.linalg.norm(np.array(self.env.robot.fingertip.pose().xyz()[:2]) - np.array(target))
            phi[index] = 1. - 4. * delta
            
        return phi
    
    def feature_dim(self):
        return len(self.target_positions)
    
    def get_w(self):
        w = np.zeros((len(self.target_positions), 1), dtype=np.float32)
        w[self.task_index, 0] = 1.0
        return w

    def _to_continuous_action(self, action):
        if np.isscalar(action):
            if self.action_dict is None:
                raise ValueError("Received discrete action but allow_discrete_actions=False")
            action_index = int(np.asarray(action).item())
            if action_index not in self.action_dict:
                raise ValueError(f"bad action {action_index}")
            a = self.action_dict[action_index]
        else:
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            if a.shape[0] != 2:
                raise ValueError(f"Expected continuous action shape (2,), got {a.shape}")
        return np.clip(a, -1.0, 1.0).astype(np.float32)


class ReacherBulletEnv(BaseBulletEnv):

    def __init__(self, target):
        self.robot = ReacherRobot(target)
        BaseBulletEnv.__init__(self, self.robot)

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def step(self, a):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # sets self.to_target_vec
        
        delta = np.linalg.norm(
            np.array(self.robot.fingertip.pose().xyz()) - np.array(self.robot.target.pose().xyz()))
        reward = 1. - 4. * delta
        self.HUD(state, a, False)
        
        return state, reward, False, {}

    def camera_adjust(self):
        x, y, z = self.robot.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)


class ReacherRobot(MJCFBasedRobot):
    TARG_LIMIT = 0.27

    def __init__(self, target):
        MJCFBasedRobot.__init__(self, 'reacher.xml', 'body0', action_dim=2, obs_dim=4)
        self.target_pos = target

    def robot_specific_reset(self, bullet_client):
        self.jdict["target_x"].reset_current_position(self.target_pos[0], 0)
        self.jdict["target_y"].reset_current_position(self.target_pos[1], 0)
        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]
        self.central_joint = self.jdict["joint0"]
        self.elbow_joint = self.jdict["joint1"]
        self.central_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.elbow_joint.reset_current_position(self.np_random.uniform(low=-3.14 / 2, high=3.14 / 2), 0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.central_joint.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
        self.elbow_joint.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))

    def calc_state(self):
        theta, self.theta_dot = self.central_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
        return np.array([
            theta,
            self.theta_dot,
            self.gamma,
            self.gamma_dot
        ])