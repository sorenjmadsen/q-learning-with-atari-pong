from collections import deque, namedtuple
import numpy as np

Experience = namedtuple("Experience", field_names=["state",    "action",      "reward", "next_state", "done"])

class ExperienceBuffer:
    def __init__(self, capacity):
        """
        Initialize a replay buffer.

        Params:
            capacity (int): Maximum size of the buffer
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(next_states), np.array(dones, dtype=bool)

class PERBuffer:
    def __init__(self, capacity, min_priority=0.01, alpha=0.5, beta=0.5, beta_growth=1.00001):
        """
        Initialize a Prioritxed Experience Replay buffer.

        Params:
            capacity (int): Maximum size of the buffer
            min_priority (float): Minimum sample priority
            alpha (float): Determines the strength of priority weighting
            beta (float): Initial beta value, determines initial strength of priority weighting
            beta_growth (float): Beta growth factor, determines how quickly correction kicks in
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.min_priority=min_priority
        self.alpha=alpha
        self.beta=beta
        self.beta_growth=beta_growth

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(self.min_priority ** self.alpha)

    def update_priorities(self, deltas, indices):
        self.beta *= self.beta_growth
        self.beta = np.min(self.beta, 1)
        for d, i in zip(deltas, indices):
            self.priorities[i] = d ** self.alpha

    def sample(self, batch_size):
        probs = np.array(self.priorities) / np.sum(self.priorities)
        weights = np.float_power(((1/len(self.buffer))*(1 / np.array(self.priorities))),self.beta)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(next_states), np.array(dones, dtype=bool),  indices, weights