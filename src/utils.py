import cv2
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium.wrappers.rendering import RecordVideo
from gymnasium.core import Wrapper
import ale_py
import datetime

import torch.optim as optim
from src.network import QNetwork
from src.agent import DoubleDQNAgent

class FireResetEnv(Wrapper):
    '''FireResentEnv: plays the Fire action and nudges the paddle to start the game'''
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, trunc, _ = self.env.step(1)
        if done or trunc:
            self.env.reset()
        obs, _, done, trunc, _ = self.env.step(2)
        if done or trunc:
            self.env.reset()
        return obs

class BufferEnv(gym.Wrapper):
    '''BufferEnv: stacks <skip> frames of observations'''
    def __init__(self, env, skip=4):
        super(BufferEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=skip)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done or trunc:
                break
        return np.concatenate(self._obs_buffer, axis=0), total_reward, done, trunc, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    '''ProcessFrame84: collapses RGB dimension and trims the image to include exclusively the game space.'''
    def __init__(self, env):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        # print(img.shape)
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        # resized_screen = resize_image(img, output_dim=(110, 84))
        # print(resized_screen.shape)
        x_t = resized_screen[18:102, :]
        x_t = (np.reshape(x_t, [84, 84]) - 87) 
        x_t = x_t.astype(np.uint8)
        x_t = np.expand_dims(x_t, 0)
        return x_t.astype(np.float32) / 255

def make_training_env(env_name):
    '''Creates the main environment object for training.'''
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode='rgb_array') 
    env = ProcessFrame84(env)
    env = BufferEnv(env)
    return FireResetEnv(env)

def make_testing_env(env_name, model_name=""):
    '''Creates the main environment object for testing. Includes video recording.'''
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode='rgb_array') 
    env = RecordVideo(env, "recording", name_prefix=f'{model_name}_{env_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}', episode_trigger=lambda x: True)
    env = ProcessFrame84(env)
    env = BufferEnv(env)
    return FireResetEnv(env)

def initialize_double_dqn_agent(train_env,
                                learning_rate, 
                                buffer, 
                                device, 
                                min_buffer_size,
                                priority_update,
                                gamma,
                                batch_size,
                                target_update,
                                criterion,
                                wandb_run):
    
    input_shape =  train_env.reset().shape
    action_size = train_env.action_space.n
    q_local = QNetwork(input_shape=input_shape, action_size=action_size)
    q_target = QNetwork(input_shape=input_shape, action_size=action_size)

    optimizer = optim.Adam(q_local.parameters(), lr=learning_rate)

    agent = DoubleDQNAgent( buffer=buffer,
                            q_local=q_local, 
                            q_target=q_target, 
                            optimizer=optimizer,
                            device=device,
                            min_buffer_size=min_buffer_size,
                            priority_update=priority_update,
                            gamma=gamma, 
                            batch_size=batch_size,
                            target_update=target_update, 
                            criterion=criterion,
                            wandb_run=wandb_run)
    return agent
