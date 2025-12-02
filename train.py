import numpy as np
import time
import torch
import torch.optim as optim
from collections import deque
from src.utils import make_training_env
from src.network import QNetwork
from src.agent import DoubleDQNAgent
from src.replay import ExperienceBuffer

def train_agent(agent, env, n_episodes=2000, max_t=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, verbose=False):
    '''
    Deep Q-Learning.
    
    Params
    ======
        agent (Agent): agent to train
        env (GymEnvironment): environment to train in
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    '''
    scores = []                        # list containing scores from each episode
    losses = []
    scores_window = deque(maxlen=100)  # last 100 scores
    losses_window = deque(maxlen=100)
    eps = eps_start                    # initialize epsilon
    start_time = time.time()
    total_steps = 0
    for i in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        loss = 0
        for t in range(max_t):
            action = agent.play(state, eps)
            # print(action)
            next_state, reward, done, trunc, _ = env.step(action)
            done = done or trunc
            loss += agent.train_step(state, action, reward, next_state, done, verbose=verbose, run=run)
            state = next_state
            score += reward
            if done:
                break 
        total_steps += t
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        losses_window.append(loss)
        losses.append(loss)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        if (i % 100 == 0) and verbose:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
            elapsed_time = time.time() - start_time
            print("Duration: ", elapsed_time)
        if np.mean(scores_window)>=19.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i-100, np.mean(scores_window)))
            break

    torch.save(agent.q_local.state_dict(), f'{type(agent)}_checkpoint.pth')
    elapsed_time = time.time() - start_time
    print("Training duration: ", elapsed_time)
    return scores

# TODO: Move to config.yaml
device = "mps"
lr = 2.5e-4
gamma = 0.99
batch_size = 32
target_update = 5_000
min_buffer_size = 5_000
buffer_capacity = 100_000
max_training_episodes = 2_500
max_steps_per_episode=10_000
eps_start=1
eps_end=0.02
eps_decay = 0.995
verbose=False

train_env = make_training_env('PongNoFrameskip-v4')

buffer = ExperienceBuffer(capacity=buffer_capacity)

input_shape =  train_env.reset().shape
action_size = train_env.action_space.n

q_local = QNetwork(input_shape=input_shape, action_size=action_size)
q_target = QNetwork(input_shape=input_shape, action_size=action_size)

optimizer = optim.Adam(q_local.parameters(), lr=lr)

action_size = train_env.action_space.n
agent = DoubleDQNAgent( buffer,
                        q_local, 
                        q_target, 
                        optimizer,
                        device,
                        min_buffer_size,
                        gamma, 
                        batch_size,
                        target_update, 
                        priority_update=None,
                        compute_weights=False,
                        wandb_run=None)

scores = train_agent(   agent, 
                        train_env, 
                        n_episodes=max_training_episodes, 
                        max_t=max_steps_per_episode,
                        eps_start=eps_start,
                        eps_end=eps_end,
                        eps_decay=eps_decay,
                        verbose=verbose)