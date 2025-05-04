""" TD3 based PIRL implementation with pytorch
        1. agentOptions
        2. pinnOptions
        3. TD3PIRLagent
        4. trainOptions
        5. train
"""

import os
import numpy as np
import random
import copy
from collections import deque # double-ended que
from tqdm import tqdm  # progress bar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# Agent Options
def agentOptions(
        DISCOUNT            = 0.99,
        ACTOR_LR            = 1e-4,
        CRITIC_LR           = 1e-3,
        REPLAY_MEMORY_SIZE  = 5_000,
        REPLAY_MEMORY_MIN   = 100,
        MINIBATCH_SIZE      = 16,
        UPDATE_TARGET_EVERY = 5, 
        UPDATE_ACTOR_EVERY  = 2,     # TD3: delayed policy updates
        TARGET_UPDATE_TAU   = 0.005, # Soft target update parameter
        NOISE_CLIP          = 0.5,   # TD3: clipped noise
        POLICY_NOISE        = 0.2,   # TD3: target policy noise
        EPSILON_INIT        = 1,
        EPSILON_DECAY       = 0.998, 
        EPSILON_MIN         = 0.01,
        RESTART_EP          = None,
        ):
    
    agentOp = {
        'DISCOUNT'           : DISCOUNT,
        'ACTOR_LR'           : ACTOR_LR,
        'CRITIC_LR'          : CRITIC_LR,
        'REPLAY_MEMORY_SIZE' : REPLAY_MEMORY_SIZE,
        'REPLAY_MEMORY_MIN'  : REPLAY_MEMORY_MIN,
        'MINIBATCH_SIZE'     : MINIBATCH_SIZE, 
        'UPDATE_TARGET_EVERY': UPDATE_TARGET_EVERY, 
        'UPDATE_ACTOR_EVERY' : UPDATE_ACTOR_EVERY,
        'TARGET_UPDATE_TAU'  : TARGET_UPDATE_TAU,
        'NOISE_CLIP'         : NOISE_CLIP,
        'POLICY_NOISE'       : POLICY_NOISE,
        'EPSILON_INIT'       : EPSILON_INIT,
        'EPSILON_DECAY'      : EPSILON_DECAY, 
        'EPSILON_MIN'        : EPSILON_MIN,
        'RESTART_EP'         : RESTART_EP
        }
    
    return agentOp

# PINN Options
def pinnOptions(
        CONVECTION_MODEL,
        DIFFUSION_MODEL,
        SAMPLING_FUN, 
        WEIGHT_PDE      = 1e-3, 
        WEIGHT_BOUNDARY = 1, 
        HESSIAN_CALC    = True,
        ):

    pinnOp = {
        'CONVECTION_MODEL': CONVECTION_MODEL,
        'DIFFUSION_MODEL' : DIFFUSION_MODEL, 
        'SAMPLING_FUN'    : SAMPLING_FUN,
        'WEIGHT_PDE'      : WEIGHT_PDE,
        'WEIGHT_BOUNDARY' : WEIGHT_BOUNDARY,
        'HESSIAN_CALC'    : HESSIAN_CALC,
        }

    return pinnOp

# ===== TD3 Networks =====
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        # Convert max_action to tensor if it's a numpy array
        if isinstance(max_action, np.ndarray):
            self.max_action = torch.FloatTensor(max_action)
        else:
            self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # Ensure max_action is on the same device as the output tensor
        max_action = self.max_action.to(state.device) if isinstance(self.max_action, torch.Tensor) else self.max_action
        return max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1

# Twin Delayed DDPG (TD3) + PIRL Agent class
class TD3PIRLagent:
    def __init__(self, state_dim, action_dim, max_action, agentOp, pinnOp):
        # Agent Options
        self.action_dim = action_dim
        self.max_action = max_action
        self.agentOp = agentOp
        self.pinnOp = pinnOp
        
        # Actor Networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=agentOp['ACTOR_LR'])
        
        # Critic Networks
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=agentOp['CRITIC_LR'])
        
        # Replay Memory
        self.replay_memory = deque(maxlen=agentOp['REPLAY_MEMORY_SIZE'])
        
        # Initialization of variables
        self.epsilon = agentOp['EPSILON_INIT'] if agentOp['RESTART_EP'] == None else max(
            self.agentOp['EPSILON_MIN'], 
            agentOp['EPSILON_INIT'] * np.power(agentOp['EPSILON_DECAY'], agentOp['RESTART_EP'])
        )
        
        self.target_update_counter = 0
        self.actor_update_counter = 0
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False
        self.total_it = 0

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def select_action(self, state):
        # Convert state to tensor
        state = torch.FloatTensor(state.reshape(1, -1))
        
        # Apply exploration noise during training
        if np.random.random() > self.epsilon:
            # Deterministic action from actor network
            with torch.no_grad():
                action = self.actor(state).cpu().data.numpy().flatten()
            
            # Add noise (only during training)
            if self.training_initialized:
                noise = np.random.normal(0, self.max_action * 0.1, size=self.action_dim)
                action = action + noise
                action = np.clip(action, -self.max_action, self.max_action)
        else:
            # Random exploration
            action = np.random.uniform(-self.max_action, self.max_action, self.action_dim)
            
        return action

    def train_step(self, experience, is_episode_done):
        # Update replay memory
        self.update_replay_memory(experience)
        
        if len(self.replay_memory) < self.agentOp['REPLAY_MEMORY_MIN']:
            return
        
        self.training_initialized = True
        self.total_it += 1
        
        # Sample mini-batch from replay memory
        batch = random.sample(self.replay_memory, self.agentOp['MINIBATCH_SIZE'])
        
        state = np.array([transition[0] for transition in batch], dtype=np.float32)
        action = np.array([transition[1] for transition in batch], dtype=np.float32)
        reward = np.array([transition[2] for transition in batch], dtype=np.float32)
        next_state = np.array([transition[3] for transition in batch], dtype=np.float32)
        done = np.array([transition[4] for transition in batch], dtype=np.float32)
        
        # Convert to tensors
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward.reshape(-1, 1))
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done.reshape(-1, 1))
        
        # ===== Update Critic =====
        with torch.no_grad():
            # Select action according to policy and add clipped noise for exploration
            noise = (torch.randn_like(action) * self.agentOp['POLICY_NOISE']).clamp(
                -self.agentOp['NOISE_CLIP'], 
                self.agentOp['NOISE_CLIP']
            )
            
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action[0], self.max_action[0])
            
            # TD3: Clipped Double-Q learning
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.agentOp['DISCOUNT'] * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # === Standard TD3 Critic Loss ===
        critic_loss_td = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # ===== PIRL: Physics Loss (PDE) =====
        # Sample PDE points
        X_PDE, X_BDini, X_BDlat = self.pinnOp['SAMPLING_FUN'](self.replay_memory)
        
        # Create tensors with gradient tracking
        X_PDE_tensor = torch.tensor(X_PDE, dtype=torch.float, requires_grad=True)
        
        # Get actions from actor for PDE points
        with torch.no_grad():
            actor_actions_pde = self.actor(X_PDE_tensor)
        
        # Calculate PDE loss
        critic_loss_pde = 0
        
        if self.pinnOp['HESSIAN_CALC']:
            # Compute convection and diffusion contributions to PDE
            # Get Q values from critic
            q1_pde, _ = self.critic(X_PDE_tensor, actor_actions_pde)
            
            # Separate time horizon component for derivative calculation
            # Assuming time_horizon is the first component of the state
            tau = X_PDE_tensor[:, 0].unsqueeze(1)
            x = X_PDE_tensor[:, 1:]
            
            # Compute partial derivative with respect to tau
            dq1_dtau = torch.autograd.grad(q1_pde.sum(), tau, create_graph=True)[0]
            
            # Compute gradient with respect to state
            dq1_dx = torch.autograd.grad(q1_pde.sum(), x, create_graph=True)[0]
            
            # Apply convection model - physical dynamics
            f = torch.tensor(np.apply_along_axis(
                self.pinnOp['CONVECTION_MODEL'], 
                1, 
                np.concatenate([X_PDE, actor_actions_pde.detach().numpy()], axis=1)
            ), dtype=torch.float32)
            
            # Convection term: ∂V/∂τ + f(s,a)·∇V
            conv_term = dq1_dtau + torch.sum(dq1_dx * f[:, 1:], dim=1, keepdim=True)
            
            # Add diffusion term if needed - this is a simplification, full implementation
            # would need Hessian calculation which is more complex
            # For now we implement only convection term
            
            # PDE loss: MSE of convection term (should be zero per HJB equation)
            critic_loss_pde = F.mse_loss(conv_term, torch.zeros_like(conv_term))
        else:
            # Simplified PDE loss without Hessian
            q1_pde, _ = self.critic(X_PDE_tensor, actor_actions_pde)
            
            # Compute gradient with respect to state
            dq1_ds = torch.autograd.grad(q1_pde.sum(), X_PDE_tensor, create_graph=True)[0]
            
            # Convection term: first component is -1 for tau, rest is dynamics
            f = torch.tensor(np.apply_along_axis(
                self.pinnOp['CONVECTION_MODEL'], 
                1, 
                np.concatenate([X_PDE, actor_actions_pde.detach().numpy()], axis=1)
            ), dtype=torch.float32)
            
            # Simple convection term
            conv_term = torch.sum(dq1_ds * f, dim=1, keepdim=True)
            
            # PDE loss: MSE of convection term (should be zero)
            critic_loss_pde = F.mse_loss(conv_term, torch.zeros_like(conv_term))
        
        # ===== PIRL: Boundary Conditions =====
        # Terminal boundary (τ = 0 and safe)
        X_BDini_tensor = torch.tensor(X_BDini, dtype=torch.float)
        with torch.no_grad():
            actions_bdini = self.actor(X_BDini_tensor)
        
        q1_bdini, _ = self.critic(X_BDini_tensor, actions_bdini)
        loss_bdini = F.mse_loss(q1_bdini, torch.ones_like(q1_bdini))
        
        # Lateral boundary (unsafe set)
        X_BDlat_tensor = torch.tensor(X_BDlat, dtype=torch.float)
        with torch.no_grad():
            actions_bdlat = self.actor(X_BDlat_tensor)
        
        q1_bdlat, _ = self.critic(X_BDlat_tensor, actions_bdlat)
        loss_bdlat = F.mse_loss(q1_bdlat, torch.zeros_like(q1_bdlat))
        
        critic_loss_boundary = loss_bdini + loss_bdlat
        
        # ===== Total Critic Loss =====
        # Weighted combination of TD loss, PDE loss, and boundary loss
        Lambda = self.pinnOp['WEIGHT_PDE']
        Mu = self.pinnOp['WEIGHT_BOUNDARY']
        critic_loss = critic_loss_td + Lambda * critic_loss_pde + Mu * critic_loss_boundary
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ===== Update Actor =====
        # Delayed policy updates
        if self.total_it % self.agentOp['UPDATE_ACTOR_EVERY'] == 0:
            # Actor loss: maximize Q-value
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            self.soft_update_target_networks()
        
        # Decay epsilon
        if is_episode_done:
            if self.epsilon > self.agentOp['EPSILON_MIN']:
                self.epsilon *= self.agentOp['EPSILON_DECAY']
                self.epsilon = max(self.agentOp['EPSILON_MIN'], self.epsilon)

    def soft_update_target_networks(self):
        # Soft update target networks
        tau = self.agentOp['TARGET_UPDATE_TAU']
        
        # Update critic targets
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Update actor target
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def get_qs(self, state):
        """Return action values for compatibility with original interface"""
        state_tensor = torch.FloatTensor(state.reshape(1, -1))
        
        # Get action from actor
        with torch.no_grad():
            action = self.actor(state_tensor)
            
        # Get Q-values for all discrete actions
        # This is a simplification to maintain compatibility with the original interface
        # We evaluate the actor's output action
        with torch.no_grad():
            q_val, _ = self.critic(state_tensor, action)
            
        # Return a tensor of appropriate size for compatibility
        # For continuous actions, we return the Q-value of the actor's chosen action,
        # and fill the rest with lower values to ensure the actor's action is selected
        result = torch.zeros(self.action_dim)
        result[0] = q_val.item()  # The first position gets the actual Q value
        
        return result

    def get_epsilon_greedy_action(self, state):
        """Return action index for compatibility with original interface"""
        # This function now returns the actual continuous action rather than an index
        return self.select_action(state)

    def load_weights(self, ckpt_dir, ckpt_idx=None):
        if not os.path.isdir(ckpt_dir):         
            raise FileNotFoundError(f"Directory '{ckpt_dir}' does not exist.")

        if not ckpt_idx or ckpt_idx == 'latest': 
            check_points = [item for item in os.listdir(ckpt_dir) if 'agent' in item]
            check_nums = np.array([int(file_name.split('-')[1]) for file_name in check_points])
            latest_ckpt = f'/agent-{check_nums.max()}'  
            ckpt_path = ckpt_dir + latest_ckpt
        else:
            ckpt_path = ckpt_dir + f'/agent-{ckpt_idx}'
            if not os.path.isfile(ckpt_path):   
                raise FileNotFoundError(f"Check point 'agent-{ckpt_idx}' does not exist.")

        checkpoint = torch.load(ckpt_path)
        self.actor.load_state_dict(checkpoint['actor_weights'])
        self.actor_target.load_state_dict(checkpoint['actor_target_weights'])
        self.critic.load_state_dict(checkpoint['critic_weights'])
        self.critic_target.load_state_dict(checkpoint['critic_target_weights'])
        self.replay_memory = checkpoint['replay_memory']
        
        print(f'Agent loaded weights stored in {ckpt_path}')
        
        return ckpt_path

# Learning Algorithm
def trainOptions(
        EPISODES      = 50, 
        LOG_DIR       = None,
        SHOW_PROGRESS = True,
        SAVE_AGENTS   = True,
        SAVE_FREQ     = 1,
        RESTART_EP    = None
        ):
    
    trainOp = {
        'EPISODES'     : EPISODES, 
        'LOG_DIR'      : LOG_DIR,
        'SHOW_PROGRESS': SHOW_PROGRESS,
        'SAVE_AGENTS'  : SAVE_AGENTS,
        'SAVE_FREQ'    : SAVE_FREQ,
        'RESTART_EP'   : RESTART_EP
        }
        
    return trainOp

def each_episode(agent, env, trainOp): 
    # Reset episodic reward and state
    episode_reward = 0
    current_state = env.reset()
    
    # For logging initial Q-value
    # Note: This is now using the critic to evaluate the actor's action
    state_tensor = torch.FloatTensor(current_state.reshape(1, -1))
    with torch.no_grad():
        action = agent.actor(state_tensor)
        q0, _ = agent.critic(state_tensor, action)
        episode_q0 = q0.item()
    
    # Iterate until episode ends
    is_done = False
    while not is_done:
        # Get action (continuous)
        action = agent.get_epsilon_greedy_action(current_state)
        
        # Make a step in the environment
        new_state, reward, is_done = env.step(action)
        episode_reward += reward
        
        # Train networks
        experience = (current_state, action, reward, new_state, is_done)
        agent.train_step(experience, is_done)
        
        # Update current state
        current_state = new_state
    
    return episode_reward, episode_q0

def train(agent, env, trainOp):
    # Log file
    if trainOp['LOG_DIR']: 
        # For training stats
        summary_writer = SummaryWriter(log_dir=trainOp['LOG_DIR'])        

    start = 1 if trainOp['RESTART_EP'] == None else trainOp['RESTART_EP']
    
    # Iterate episodes
    if trainOp['SHOW_PROGRESS']:     
        iterator = tqdm(range(start+1, trainOp['EPISODES'] + 1), ascii=True, unit='episodes')
    else:
        iterator = range(start+1, trainOp['EPISODES'] + 1)

    for episode in iterator:
        ep_reward, ep_q0 = each_episode(agent, env, trainOp)

        if trainOp['LOG_DIR']: 
            summary_writer.add_scalar("Episode Reward", ep_reward, episode)
            summary_writer.add_scalar("Episode Q0", ep_q0, episode)
            summary_writer.flush()

            if trainOp['SAVE_AGENTS'] and episode % trainOp['SAVE_FREQ'] == 0:
                ckpt_path = trainOp['LOG_DIR'] + f'/agent-{episode}'
                torch.save({
                    'actor_weights': agent.actor.state_dict(),
                    'actor_target_weights': agent.actor_target.state_dict(),
                    'critic_weights': agent.critic.state_dict(),
                    'critic_target_weights': agent.critic_target.state_dict(),
                    'replay_memory': agent.replay_memory
                }, ckpt_path)
                
    return
