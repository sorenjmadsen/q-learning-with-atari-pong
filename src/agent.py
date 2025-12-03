import numpy as np
import torch
import torch.nn.functional as F
from .replay import Experience, ExperienceBuffer, PERBuffer

class DoubleDQNAgent:
    ''' A Double DQN that interacts with and learns from the environment. '''
    def __init__(self, 
                 buffer,
                 q_local, 
                 q_target, 
                 optimizer,
                 device,
                 min_buffer_size,
                 priority_update,
                 gamma, 
                 batch_size,
                 target_update, 
                 criterion,
                 wandb_run=None):
        ''' 
        Initialize an Agent object.
        
        Params
        ======
            buffer (ExperienceBuffer or PERBuffer): replay buffer (regular or prioritized)
            q_local (Module): Local Q network
            q_target (Module): Target Q network, must be the same structure as q_local
            optimizer (Optimizer): Optimizer to use for training
            device (str): Device for training
            min_buffer_size (int): Minimum size of buffer to start training (default: 5,000)
            priority_update (int): Number of steps to take between priority weighting updates (optional, default: 20)
            gamma (float): Reward discount (default: 0.99) 
            batch_size (int): Number of samples per batch during train_samples (default: 32)
            target_update (int): Number of steps to take between target network updates (default: 5000) 
            criterion (Loss): Loss function to optimize (default: SmoothL1Loss())
            wandb_run: Track your experiment with a Weights & Biases run (default: None)
        '''
        self.device = device
        self.batch_size = batch_size
        self.target_update = target_update
        self.priority_update = priority_update
        self.min_buffer_size = min_buffer_size

        self.gamma = gamma

        # Q Network Preparation
        self.q_local = q_local.to(device)
        self.q_target = q_target.to(device)
        self.update_target_network()
        self.q_target.eval()

        self.optimizer = optimizer
        self.criterion = criterion

        # Replay memory
        self.buffer = buffer
        self.compute_weights = isinstance(self.buffer, PERBuffer)
        if self.compute_weights:
            print('Using PERBuffer')
            self.update_priority_step = 0

        # Update params
        self.update_network_step = 0
        self.noop = 0
        self.run = wandb_run
    
    def train_step(self, state, action, reward, next_state, done, verbose, run=None):
        # Add experience to the buffer
        self.buffer.add(Experience(state, action, reward, next_state, done))
        loss = 0

        # If enough samples are available in memory, get random subset and learn
        if len(self.buffer) >=  self.min_buffer_size:
            # Update counters
            self.update_network_step = (self.update_network_step + 1) % self.target_update
            if self.compute_weights:
                self.update_priority_step = (self.update_priority_step + 1) % self.priority_update

            # Train on batch and (possibly update priority weights)
            samples = self.buffer.sample(self.batch_size)
            loss = self.train_samples(samples)

            # Update target network
            if self.update_network_step == 0:
                self.update_target_network()

        return loss
    
    def train_samples(self, sample):
        ''' Update value parameters using given batch of experience tuples. '''
        if self.compute_weights:
            states, actions, rewards, next_states, dones, indices, weights = sample
            print(rewards.shape, next_states.shape, dones.shape)
        else:
            states, actions, rewards, next_states, dones = sample

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(rewards).float().to(self.device).unsqueeze(1)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).to(self.device).unsqueeze(1)
        
       
        with torch.no_grad():
            next_actions = self.q_local(next_states).argmax(dim=1, keepdim=True)
            q_targets = self.q_target(next_states).gather(1, next_actions) # Target network values across local network predicted actions

        expected_values = rewards + self.gamma * q_targets*(~dones)

        self.optimizer.zero_grad()
        self.q_local.train()
        output = self.q_local(states.float())

        if self.run:
            q_max = output.max().item()
            q_mean = output.mean()
            target_mean = (q_targets*(~dones)).mean()
            ev_mean = expected_values.mean()

        output = output.gather(1, actions)

        if self.compute_weights:
            # Convert weights to torch tensor
            weights_tensor = torch.from_numpy(weights).float().to(self.device).unsqueeze(1)

            # Apply importance sampling weights to individual losses
            elementwise_loss = F.smooth_l1_loss(output, expected_values, reduction='none')
            loss = (elementwise_loss * weights_tensor).mean()
        else:
            loss = self.criterion(output, expected_values)
        
        loss.backward()

        # For additional stability
        torch.nn.utils.clip_grad_norm_(self.q_local.parameters(), 5.0)
        self.optimizer.step()

        # ------------------- update priorities ------------------- #
        if self.compute_weights and (self.update_priority_step == 0):
            delta = abs(expected_values - output.detach()).squeeze(1).cpu().numpy()    
            self.buffer.update_priorities(delta, indices)  

        if self.run:
            if self.compute_weights:
                self.run.log({
                    "loss"   : loss.item(),
                    "local_q_max"  : q_max,
                    "local_q_mean" : q_mean,
                    "target_q_mean": target_mean.cpu().numpy(),
                    "rewards": rewards.mean().cpu().numpy(),
                    "expected_values": ev_mean.cpu().numpy(),
                    "per_delta_mean": delta.mean(),
                    "per_delta_max": delta.max()
                })
            else:
                self.run.log({
                    "loss"   : loss.item(),
                    "local_q_max"  : q_max,
                    "local_q_mean" : q_mean,
                    "target_q_mean": target_mean.cpu().numpy(),
                    "rewards": rewards.mean().cpu().numpy(),
                    "expected_values": ev_mean.cpu().numpy(),
                })
            
        return loss.item()

    def play(self, state, eps=0.):
        ''' 
        Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_local.eval()
        with torch.no_grad():
            _, action = torch.max(self.q_local(state), dim=1)

        # Epsilon-greedy action selection
        x = np.random.random()
        if x > eps:
            return int(action.item())
        else:
            return np.random.choice([0, 1, 2, 3, 4, 5])

   
    def update_target_network(self):
        ''' Update target model parameters. '''
        self.q_target.load_state_dict(self.q_local.state_dict())


class InferenceQAgent:
    def __init__(self, 
                 q_network,
                 device):
        ''' 
        Initialize an InferenceAgent object.
        
        Params
        ======
            q_local (Module): Trained Q network
            device: Device for inference
        '''
        self.q_network = q_network
        self.device = device
        self.q_network.to(device)
        self.q_network.eval()


    def play(self, state, eps=0.):
        ''' 
        Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, action = torch.max(self.q_network(state), dim=1)

        # Epsilon-greedy action selection
        x = np.random.random()
        if x > eps:
            return int(action.item())
        else:
            return np.random.choice([0, 1, 2, 3, 4, 5])
