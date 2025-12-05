import numpy as np
import time
import torch
from collections import deque
from pathlib import Path
from omegaconf import OmegaConf
from src.utils import make_training_env, initialize_double_dqn_agent

from torch.nn import SmoothL1Loss
from src.replay import ExperienceBuffer, PERBuffer
import datetime
import os


def train_agent(agent, env, n_episodes=2000, max_t=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, verbose=False, run=None):
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

        if verbose: 
            print('\rEpisode {}\tAverage Score: {:.2f}\tLoss: {:.4f}\tTotal Steps: {}'.format(i, np.mean(scores_window), loss / t, total_steps))
        if (i % 100 == 0) and verbose:
            elapsed_time = time.time() - start_time
            print("\tDuration: ", elapsed_time)

        if np.mean(scores_window)>=19.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))
            break
    return agent, scores

if __name__ == '__main__':
    CONFIG_PATH = Path(__file__).with_name('config.yaml')
    file_config = OmegaConf.load(CONFIG_PATH) if CONFIG_PATH.exists() else OmegaConf.create()
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(file_config, cli_config)

    training_config = config.get('training', {})
    epsilon_config = training_config.get('epsilon', {})
    buffer_config = training_config.get('buffer', {})
    wandb_config = training_config.get('wandb_config', {})

    architecture = training_config.get('architecture', 'DoubleDQN')
    environment_id = training_config.get('environment_id', 'PongNoFrameskip-v4')
    device = training_config.get('device', 'cpu')
    lr = training_config.get('learning_rate', 2.5e-4)
    gamma = training_config.get('gamma', 0.99)
    batch_size = training_config.get('batch_size', 32)
    target_update = training_config.get('target_update', 5_000)
    buffer_type = training_config.get('buffer_type', 'ExperienceReplay')

    min_buffer_size = buffer_config.get('min_buffer_size', 5_000)
    buffer_capacity = buffer_config.get('buffer_capacity', 100_000)
    per_priority_update = -1 # Won't be triggered for an ExperienceBuffer
    if buffer_type == 'PERBuffer':
        per_config = training_config.get('buffer', {})
        per_min_priority = per_config.get('min_priority', 0.01)
        per_alpha = per_config.get('alpha', 0.5)
        per_beta = per_config.get('beta', 0.5)
        per_beta_growth = per_config.get('beta_growth', 1.00001)
        per_priority_update = per_config.get('per_update', 20)

    max_training_episodes = training_config.get('max_episodes', 2_000)
    max_steps_per_episode = training_config.get('max_steps_per_episode', 10_000)

    eps_start = epsilon_config.get('start', 1.0)
    eps_end = epsilon_config.get('end', 0.01)
    eps_decay = epsilon_config.get('decay', 0.995)
    verbose = training_config.get('verbose', False)
    use_wandb = training_config.get('use_wandb', False)

    criterion=SmoothL1Loss() # used in the Deepmind paper, less sensitive to outliers

    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('cuda is requested but no cuda available')
    elif device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is requested but no MPS available')

    assert(buffer_type == 'ExperienceBuffer' or buffer_type == 'PERBuffer')

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            entity=wandb_config.get('entity_name', ''),
            # Set the wandb project where this run will be logged.
            project=f"{architecture}-{buffer_type}-{environment_id}",
            # Track hyperparameters and run metadata.
            config=training_config
        )

    train_env = make_training_env(environment_id)

    if buffer_type == 'ExperienceBuffer':
        buffer = ExperienceBuffer(capacity=buffer_capacity)
    elif buffer_type == 'PERBuffer':
        buffer = PERBuffer(capacity=buffer_capacity,
                           min_priority=per_min_priority,
                           alpha=per_alpha,
                           beta=per_beta,
                           beta_growth=per_beta_growth)
        
    agent = initialize_double_dqn_agent(train_env=train_env,
                                        learning_rate=lr, 
                                        buffer=buffer, 
                                        device=device, 
                                        min_buffer_size=min_buffer_size,
                                        priority_update=per_priority_update,
                                        gamma=gamma,
                                        batch_size=batch_size,
                                        target_update=target_update,
                                        criterion=criterion,
                                        wandb_run=wandb_run)
    
    start_time = time.time()
    print('=======================================')
    print(f'Training agent on {environment_id}...')
    print(f'|--- episodes:  {max_training_episodes}')
    print(f'|--- max steps: {max_steps_per_episode}')
    print(f'|--- device:    {device}')
    print('=======================================')
    agent, scores = train_agent(agent, 
                                train_env, 
                                n_episodes=max_training_episodes, 
                                max_t=max_steps_per_episode,
                                eps_start=eps_start,
                                eps_end=eps_end,
                                eps_decay=eps_decay,
                                verbose=verbose)
    elapsed_time = time.time() - start_time
    agent.q_local.to('cpu') # Had some serialization issues on non-MacOS devices (for mps backend) if I don't move back to CPU first
    torch.save(agent.q_local.state_dict(), os.path.join('checkpoints', f'{architecture}-{buffer_type}-{environment_id}-final-_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.pth'))
    print("Training duration: ", elapsed_time)
