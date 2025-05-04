# -*- coding: utf-8 -*-
"""
TD3-PIRL implementation for safe drifting with Lagrangian method
@author: based on Hikaru Hoshino's work
"""
# general packages
import numpy as np
import random
from datetime import datetime
import argparse 
import torch

from torch.optim import Adam

# TD3-PIRL agent
from rl_agent.TD3_PIRL_torch import TD3PIRLagent, agentOptions, train, trainOptions, pinnOptions
from rl_env.carla_env import CarEnv, spawn_train_map_c_north_east, map_c_before_corner

##############################################################
# carla environment
class Env(CarEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def reset(self):
        carla_state = super().reset()
        horizon = np.random.uniform(4.0, 6.0)
        self.state = np.array(list(carla_state) + [horizon])        
        return self.state

    def step(self, action):
        """
        Modified to handle continuous actions from TD3
        Takes direct physical action values within proper ranges
        """
        # Extract steering and throttle from continuous action
        steering = action[0]  # Should be in range [-0.8, 0.8]
        throttle = action[1]  # Should be in range [0.6, 1.0]
        
        # Find closest discrete indices in action pools
        steering_idx = np.argmin(np.abs(np.array(self.step_S_pool) - steering))
        throttle_idx = np.argmin(np.abs(np.array(self.step_T_pool) - throttle))
        
        # Calculate action index
        action_idx = throttle_idx * len(self.step_S_pool) + steering_idx
        
        # Make a step with the discrete action
        new_veh_state, reward, done = super().step(action_idx)
        
        # Update horizon
        horizon = self.state[-1] - self.time_step
        new_state = np.array(list(new_veh_state) + [horizon])
        self.state = new_state

        # Rewrite "reward" and "done" based on horizon
        if horizon <= 0:
            done = True
            reward = 1
        
        return new_state, reward, done        

###############################################################################
# Physics information
def convection_model(s_and_act):
    """Physics model for drift dynamics with continuous actions"""
    # Parse state and action
    s = s_and_act[:-2]  # State without action
    action = s_and_act[-2:]  # Last two elements are continuous actions [steering, throttle]

    x = s[:-1]
    vx = x[0]            # m/s
    beta = x[1]*(3.14/180)  # deg -> rad
    vy = vx*np.tan(beta)
    omega = x[2]*(3.14/180)  # deg/s -> rad/s  
    psi = x[4]*(3.14/180)  # deg -> rad              

    # Extract continuous actions
    steer = action[0]  # Should be in range [-0.8, 0.8]
    throttle = action[1]  # Should be in range [0.6, 1.0]
    
    # Scale steering to physical limits (±70 degrees)
    steer = steer * 70 * (3.14/180)  # [-0.8,0.8] -> [-56deg,56deg] -> [-0.98rad, 0.98rad]
    
    # Parameters
    lf = 1.34
    lr = 1.3
    mass = 1265
    Iz = 2093
    Bf = 5.579
    Cf = 1.2
    Df = 16000
    Br = 7.579
    Cr = 1.2
    Dr = 16000

    Cm1 = 550*(3.45*0.919)/(0.34)
    Cm2 = 0 
    Cr0 = 50.
    Cr2 = 0.5

    # Model equation
    dxdt = np.zeros(15) 
    Frx = (Cm1-Cm2*vx)*throttle - Cr0 - Cr2*(vx**2)
    alphaf = steer - np.arctan2((lf*omega + vy), abs(vx))
    alphar = np.arctan2((lr*omega - vy), abs(vx))
    Ffy = Df * np.sin(Cf * np.arctan(Bf * alphaf))
    Fry = Dr * np.sin(Cr * np.arctan(Br * alphar))

    dxdt[0] = 1/mass * (Frx - Ffy*np.sin(steer)) + vy*omega    # vx
    dxdt[1] = 1/(mass*vx) * (Fry + Ffy*np.cos(steer)) - omega  # beta
    dxdt[2] = 1/Iz * (Ffy*lf*np.cos(steer) - Fry*lr)           # omega
    dxdt[3] = vy*np.cos(psi) + vx*np.sin(psi)                  # lat_error
    dxdt[4] = omega                                            # psi

    dsdt = np.concatenate([dxdt, np.array([-1])])
    
    return dsdt


def diffusion_model(x_and_act):
    # Including both state and continuous actions
    diagonals = np.concatenate([0.1*np.ones(5), 0*np.ones(10), np.array([0])])
    sig = np.diag(diagonals)
    diff = np.matmul(sig, sig.T)
 
    return diff

def sample_for_pinn(replay_memory):
    n_dim = 15 + 1
    T = 5
    Emax = 8
    x_vehicle_max = np.concatenate([np.array([35,-30, 150]+[Emax, 60]), np.ones(10)*3])
    x_vehicle_min = np.concatenate([np.array([25,-15, 0]+[-Emax,-60]), -np.ones(10)*3])

    # Interior points    
    nPDE = 32
    x_max = np.array(list(x_vehicle_max) + [T])
    x_min = np.array(list(x_vehicle_min) + [0])
    X_PDE = x_min + (x_max - x_min) * np.random.rand(nPDE, n_dim)
    sample_state = [replay_memory[np.random.randint(0,len(replay_memory))][0] for i in range(nPDE)]
    X_PDE[:,5:-1] = np.asarray(sample_state)[:, 5:-1]
    assert X_PDE.shape == (nPDE, n_dim)

    # Terminal boundary (at T=0 and safe)
    nBDini = 32
    x_max = np.array(list(x_vehicle_max) + [0])
    x_min = np.array(list(x_vehicle_min) + [0])
    X_BD_TERM = x_min + (x_max - x_min) * np.random.rand(nBDini, n_dim)
    sample_state = [replay_memory[np.random.randint(0,len(replay_memory))][0] for i in range(nBDini)]
    X_BD_TERM[:,5:15] = np.asarray(sample_state)[:, 5:15]
    assert X_BD_TERM.shape == (nBDini, n_dim)

    # Lateral boundary (unsafe set)        
    nBDsafe = 32
    x_max = np.array(list(x_vehicle_max) + [T])
    x_min = np.array(list(x_vehicle_min) + [0])
    X_BD_LAT = x_min + (x_max - x_min) * np.random.rand(nBDsafe, n_dim)
    X_BD_LAT[:,3] = np.random.choice([-Emax, Emax], size=nBDsafe)    
    sample_state = [replay_memory[np.random.randint(0,len(replay_memory))][0] for i in range(nBDsafe)]
    X_BD_LAT[:,5:15] = np.asarray(sample_state)[:, 5:15]
    X_BD_LAT[:,3] = np.random.choice([-Emax, Emax], size=nBDsafe)
    assert X_BD_LAT.shape == (nBDsafe, n_dim)
    
    return X_PDE, X_BD_TERM, X_BD_LAT
    
################################################################################################
# Main
if __name__ == '__main__':
    """
    run carla by: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=4000 &
    """    
    ######################################
    # Settings
    ######################################
    carla_port = 4000
    time_step = 0.05
    map_train = "./maps/train.xodr"

    #######################################
    # arg parse
    #######################################
    parser = argparse.ArgumentParser() 
    parser.add_argument('--port', type=int, default=4000, help='CARLA server port')
    parser.add_argument('--host', type=str, default="localhost", help='CARLA server hostname')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--lambda_lr', type=float, default=1e-4, help='Lambda learning rate')
    parser.add_argument('--pde_epsilon', type=float, default=0.2, help='PDE constraint threshold')
    parser.add_argument('--lambda_max', type=float, default=5.0, help='Maximum lambda value')
    parser.add_argument('--decay_start', type=int, default=0, help='Constraint decay start episode')
    parser.add_argument('--decay_end', type=int, default=30000, help='Constraint decay end episode')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU acceleration')
    args = parser.parse_args() 
    
    if args.port:
        carla_port = args.port
    
    carla_host = args.host
    
    ###############################################################
    # Environment
    ###############################################################
    # vehicle state initialization
    def vehicle_reset_method():
        # position and angle
        x_loc = 0
        y_loc = 0 
        psi_loc = 0
        # velocity and yaw rate
        vx = 30
        vy = -vx * np.random.uniform(np.tan(20/180*3.14), np.tan(25/180*3.14))
        yaw_rate = np.random.uniform(50, 70)
        
        # It must return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]
        return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]

    # Spectator coordinate
    spec_mapC_NorthEast = {'x':-965, 'y':185, 'z':15, 'pitch':-45, 'yaw':120, 'roll':0} 

    # Create environment (not passing host parameter)
    env = Env(port=carla_port,
              time_step=time_step,
              custom_map_path=map_train,
              actor_filter='vehicle.audi.tt',  
              spawn_method=map_c_before_corner,
              vehicle_reset=vehicle_reset_method, 
              waypoint_itvl=3.0,
              spectator_init=spec_mapC_NorthEast,
              spectator_reset=False)
    
    state_dim = len(env.reset())
    action_dim = 2  # [steering, throttle]
    max_action = np.array([1.0, 1.0])  # Max values for normalizing actor output

    ###########################################################################
    # TD3-PIRL Options
    learning_rate = args.lr
    
    agentOp = agentOptions(
        DISCOUNT=1,
        ACTOR_LR=learning_rate,
        CRITIC_LR=learning_rate*2,            # Critics often use higher learning rates
        LAMBDA_LR=args.lambda_lr,             # Learning rate for Lagrangian multiplier
        REPLAY_MEMORY_SIZE=10000,
        REPLAY_MEMORY_MIN=1000,
        MINIBATCH_SIZE=256,
        UPDATE_TARGET_EVERY=5,
        UPDATE_ACTOR_EVERY=2,                 # Delayed policy updates (TD3)
        TARGET_UPDATE_TAU=0.005,
        NOISE_CLIP=0.5,
        POLICY_NOISE=0.2,
        EPSILON_DECAY=0.9998,
        EPSILON_MIN=0.001,
        PDE_EPSILON=args.pde_epsilon,         # Threshold for PDE constraint
        USE_GPU=not args.no_gpu,              # Use GPU unless disabled by flag
        CONSTRAINT_DECAY_START=args.decay_start,
        CONSTRAINT_DECAY_END=args.decay_end,
        LAMBDA_MAX=args.lambda_max,           # Maximum lambda value
        LOG_LAGRANGIAN_COMPONENTS=True,       # Log the components of the Lagrangian
    )
    
    pinnOp = pinnOptions(
        CONVECTION_MODEL=convection_model,
        DIFFUSION_MODEL=diffusion_model,
        SAMPLING_FUN=sample_for_pinn,
        WEIGHT_PDE=1e-4,
        WEIGHT_BOUNDARY=1,
        HESSIAN_CALC=False,
    )
    
    # Create TD3-PIRL agent
    agent = TD3PIRLagent(state_dim, action_dim, max_action, agentOp, pinnOp)

    ######################################
    # Training option
    restart = False

    if restart:
        LOG_DIR = "logs/MapC/TD3_Lagrangian_04291633/"  # Set log dir
        ckp_path = agent.load_weights(LOG_DIR, ckpt_idx='latest')
        current_ep = int(ckp_path.split('-')[-1].split('.')[0])
        print(f"Restarting from episode {current_ep}")
    else:
        LOG_DIR = f'logs/MapC/TD3_Lagrangian_'+datetime.now().strftime('%m%d%H%M')
        current_ep = None

    """
    $ tensorboard --logdir logs/...
    """
    
    trainOp = trainOptions(
        EPISODES=30_000,
        SHOW_PROGRESS=True,
        LOG_DIR=LOG_DIR,
        SAVE_AGENTS=True,
        SAVE_FREQ=500,
        RESTART_EP=current_ep
    )
    
    if current_ep is not None:
        agentOp['RESTART_EP'] = current_ep

    ######################################
    # Train 
    ######################################
    # Set seeds for reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    # Print training configuration
    print("="*80)
    print("Training TD3-PIRL with Lagrangian method")
    print(f"Using device: {agent.device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Lambda learning rate: {args.lambda_lr}")
    print(f"PDE epsilon: {args.pde_epsilon}")
    print(f"Lambda max: {args.lambda_max}")
    print(f"Constraint decay: {args.decay_start} to {args.decay_end}")
    print(f"Saving logs to: {LOG_DIR}")
    print("="*80)

    try:
        train(agent, env, trainOp)
     
    except KeyboardInterrupt:
        print('\nCancelled by user - training.py.')

    finally:
        if 'env' in locals():
            env.destroy()
