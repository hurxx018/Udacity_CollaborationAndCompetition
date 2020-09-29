from collections import deque, namedtuple
import copy
import itertools
import random


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from ddpg_model import Actor, Critic, initialize_weights


BUFFER_SIZE = int(2e5)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-2         # learning rate of the actor 
LR_CRITIC = 1e-2        # learning rate of the critic
WEIGHT_DECAY = 0.00001  # L2 weight decay

UPDATE_EVERY = 8       # how often to update the network
N_LEARNING = 2

# OS Noise parameters
THETA = 0.1*1
SIGMA = 0.05*1

THETA_MIN = 0.0000001
SIGMA_MIN = 0.0000001
DECAY_FACTOR_S = 0.9999
DECAY_FACTOR_T = 0.9999


class Agent():

    def __init__(
        self,
        n_state,
        n_action,
        random_seed,
        device = "cpu"
        ):
        """Initialize an Agent object.
        
        Params
        ------
            n_state : int
                dimension of each state
            n_action : int
                dimension of each action
            random_seed : int
                random seed
            device :
                which device is used, cpu or cuda.
        """
        self.n_state = n_state
        self.n_action = n_action
        self.random_seed = np.random.seed(random_seed)
        self.device = device

        # Networks for the first agent
        # Local Actor, Local Critic, Target Actor, Target Critic
        self.actor_local1 = Actor(self.n_state, self.n_action, self.random_seed).to(self.device)
        self.actor_local1.apply(initialize_weights)
        self.critic_local1 = Critic(self.n_state, self.n_action, self.random_seed).to(self.device)
        self.critic_local1.apply(initialize_weights)
        self.actor_target1 = Actor(self.n_state, self.n_action, self.random_seed).to(self.device)
        self.actor_target1.apply(initialize_weights)
        self.actor_target1.eval()
        self.critic_target1 = Critic(self.n_state, self.n_action, self.random_seed).to(self.device)
        self.critic_target1.apply(initialize_weights)
        self.critic_target1.eval()


        # Networks for the second agent
        # Local Actor, Local Critic, Target Actor, Target Critic
        self.actor_local2 = Actor(self.n_state, self.n_action, self.random_seed).to(self.device)
        self.actor_local2.apply(initialize_weights)
        self.critic_local2 = Critic(self.n_state, self.n_action, self.random_seed).to(self.device)
        self.critic_local2.apply(initialize_weights)
        self.actor_target2 = Actor(self.n_state, self.n_action, self.random_seed).to(self.device)
        self.actor_target2.apply(initialize_weights)
        self.actor_target2.eval()
        self.critic_target2 = Critic(self.n_state, self.n_action, self.random_seed).to(self.device)
        self.actor_target2.apply(initialize_weights)
        self.critic_target2.eval()

        # optimizers
        actor_params = [self.actor_local1.parameters(), self.actor_local2.parameters()]
        self.actor_optimizer = optim.Adam(itertools.chain(*actor_params), lr = LR_ACTOR)
        critic_params = [self.critic_local1.parameters(), self.critic_local2.parameters()]
        self.critic_optimizer = optim.Adam(itertools.chain(*critic_params), lr = LR_CRITIC, weight_decay = WEIGHT_DECAY)


        # Noise process
        self.noise = OUNoise(n_action*2, random_seed + 1, mu=0., theta=THETA, sigma=SIGMA)

        # Replay Buffer
        self.memory = ReplayBuffer(n_action, BUFFER_SIZE, BATCH_SIZE, random_seed + 2, self.device)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(
        self, 
        state, 
        action, 
        reward, 
        next_state, 
        done
        ):
        pass
       # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # Learn, if enough samples are available in memory
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(N_LEARNING):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)


    def act(self, state, add_noise = True):

        state0 = torch.from_numpy(state[0]).unsqueeze(dim=0).float().to(self.device)
        state1 = torch.from_numpy(state[1]).unsqueeze(dim=0).float().to(self.device)
        self.actor_local1.eval()
        self.actor_local2.eval()
        with torch.no_grad():
            action0 = self.actor_local1(state0).cpu().data.numpy()
            action1 = self.actor_local2(state1).cpu().data.numpy()

        action = np.vstack([action0, action1])
        self.actor_local1.train()
        self.actor_local2.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(
        self, 
        experiences, 
        gamma
        ):
        states1, actions1, rewards1, next_states1, dones1 = experiences[0]
        states2, actions2, rewards2, next_states2, dones2 = experiences[1]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next1 = self.actor_target1(next_states1)
        Q_targets_next1 = self.critic_target1(next_states1, actions_next1)
        # Compute Q targets for current states (y_i)
        Q_targets1 = rewards1 + (gamma * Q_targets_next1 * (1 - dones1))
        # Compute critic loss
        Q_expected1 = self.critic_local1(states1, actions1)

        actions_next2 = self.actor_target2(next_states2)
        Q_targets_next2 = self.critic_target2(next_states2, actions_next2)
        # Compute Q targets for current states (y_i)
        Q_targets2 = rewards2 + (gamma * Q_targets_next2 * (1 - dones2))
        # Compute critic loss
        Q_expected2 = self.critic_local2(states2, actions2)

        critic_loss = F.mse_loss(Q_expected1, Q_targets1) + F.mse_loss(Q_expected2, Q_targets2)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local1.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic_local2.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred1 = self.actor_local1(states1)
        actions_pred2 = self.actor_local2(states2)
        actor_loss = -self.critic_local1(states1, actions_pred1).mean() - self.critic_local2(states2, actions_pred2).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local1, self.critic_target1, TAU)
        self.soft_update(self.actor_local1, self.actor_target1, TAU)    
        self.soft_update(self.critic_local2, self.critic_target2, TAU)
        self.soft_update(self.actor_local2, self.actor_target2, TAU)    

    def soft_update(
        self, 
        local_model, 
        target_model, 
        tau
        ):
        """Soft update model parameters

        Arguments
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.05, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones((2, size//2))
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        self.sigma = copy.copy(max(self.sigma*DECAY_FACTOR_S, SIGMA_MIN))
        self.theta = copy.copy(max(self.theta*DECAY_FACTOR_T, THETA_MIN))

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + np.random.normal(0, self.sigma, x.shape)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, n_action, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.n_action = n_action
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states1, actions1, rewards1, next_states1, dones1 = [], [], [], [], []
        states2, actions2, rewards2, next_states2, dones2 = [], [], [], [], []
        for e in experiences:
            if e is not None:
                states1.append(e.state[0])
                actions1.append(e.action[0])
                rewards1.append(e.reward[0])
                next_states1.append(e.next_state[0])
                dones1.append(e.done)
                states2.append(e.state[1])
                actions2.append(e.action[1])
                rewards2.append(e.reward[1])
                next_states2.append(e.next_state[1])
                dones2.append(e.done)


        states1 = torch.from_numpy(np.vstack(states1)).float().to(self.device)
        actions1 = torch.from_numpy(np.vstack(actions1)).float().to(self.device)
        r_tmp1 = np.vstack(rewards1)
        m0 = r_tmp1.mean()
        s0 = r_tmp1.std() + 1e-10
        rewards1 = torch.from_numpy((r_tmp1 - m0)/s0).float().to(self.device)
        next_states1 = torch.from_numpy(np.vstack(next_states1)).float().to(self.device)
        dones1 = torch.from_numpy(np.vstack(dones1).astype(np.uint8)).float().to(self.device)

        states2 = torch.from_numpy(np.vstack(states2)).float().to(self.device)
        actions2 = torch.from_numpy(np.vstack(actions2)).float().to(self.device)
        r_tmp2 = np.vstack(rewards2)
        m0 = r_tmp2.mean()
        s0 = r_tmp2.std() + 1e-10
        rewards2 = torch.from_numpy((r_tmp2 - m0)/s0).float().to(self.device)
        next_states2 = torch.from_numpy(np.vstack(next_states2)).float().to(self.device)
        dones2 = torch.from_numpy(np.vstack(dones2).astype(np.uint8)).float().to(self.device)

        return (states1, actions1, rewards1, next_states1, dones1), (states2, actions2, rewards2, next_states2, dones2)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)