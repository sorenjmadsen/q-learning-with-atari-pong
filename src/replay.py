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
    def __init__(self, capacity, min_priority=0.01):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.priorities = []
        self.min_priority=0.01
        self.alpha=0.5
        self.beta=0.1
        self.beta_growth=1.00001

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)
        
        if len(self.priorities) < self.capacity:
            self.priorities.append(self.min_priority ** self.alpha)
        else:
            self.priorities.pop(0)
            self.priorities.append(self.min_priority ** self.alpha)

    def update_priorities(self, deltas, indices):
        self.beta *= self.beta_growth
        self.beta = min(self.beta, 1)

        for d, i in zip(deltas, indices):
            # Convert priority to a scalar float
            priority = float(abs(d) + 1e-6) ** self.alpha
            self.priorities[i] = priority


    def sample(self, batch_size):
        probs = np.array(self.priorities) / np.sum(self.priorities)
        preweights =((1/len(self.buffer))*(1 / np.array(self.priorities)))
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        weights = np.float_power(preweights,self.beta)[indices]
        weights = weights / (weights.max() + 1e-12)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(next_states), np.array(dones, dtype=bool),  indices, weights