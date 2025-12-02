import cv2
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium.wrappers.rendering import RecordVideo
from gymnasium.core import Wrapper

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
    env = gym.make(env_name, render_mode='rgb_array') 
    env = ProcessFrame84(env)
    env = BufferEnv(env)
    return FireResetEnv(env)

def make_testing_env(env_name, model_name=""):
    '''Creates the main environment object for testing. Includes video recording.'''
    env = gym.make(env_name, render_mode='rgb_array') 
    env = RecordVideo(env, "recording", name_prefix=f'DQN-{env_name}', episode_trigger=lambda x: True)
    env = ProcessFrame84(env)
    env = BufferEnv(env)
    return FireResetEnv(env)