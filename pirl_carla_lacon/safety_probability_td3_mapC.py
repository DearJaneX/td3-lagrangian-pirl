
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

sys.path.append(os.pardir)
sys.path.append('.')
from TD3_PIRL_torch import TD3PIRLagent, agentOptions, pinnOptions
from rl_env.continuous_carla_env import ContinuousCarEnv
from training_td3_pirl import Env, convection_model, diffusion_model, sample_for_pinn, map_c_before_corner

###########################################################
# Settings
###########################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default="logs/MapC/TD3_Cont_Lagrangian_04181330")
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default="localhost")
    parser.add_argument('--use_lagrangian', action='store_true')
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()
    return args


def load_agent(env, log_dir, use_lagrangian=False, use_gpu=True):
    state_dim = len(env.reset())
    action_dim = 2  # [steering, throttle]
    max_action = np.array([1.0, 1.0])  
    agentOp = agentOptions(USE_LAGRANGIAN=use_lagrangian, USE_GPU=use_gpu)
    pinnOp = pinnOptions(convection_model, diffusion_model, sample_for_pinn)    
    agent = TD3PIRLagent(state_dim, action_dim, max_action, agentOp, pinnOp)
    agent.load_weights(log_dir, ckpt_idx='19000')

    return agent


def contour_plot(x, y, z, key=None, filename=None, use_lagrangian=False):
    #(x, y) = (e, psi)/ (e, vx)
    x, y = np.meshgrid(x, y)
    x = x.transpose()
    y = y.transpose()
    
    method_name = "TD3-PIRL with Continuous Actions" + (" and Lagrangian" if use_lagrangian else "")
    
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x, y, z, cmap='viridis')
    cbar = plt.colorbar(contour)  
    cbar.set_label('Safety Probability')
    
    if not key:
        plt.xlabel(r'Slip angle $\beta$ [deg]')
        plt.ylabel('Yaw rate $r$ [deg/s]')
        plt.title(f'Safety Probability Visualization ({method_name})')
    elif key[0] == "e" and key[1] == 'psi':
        plt.xlabel('Lateral error $e$ [m]')
        plt.ylabel(r'Heading error $\psi$ [rad]')
        plt.title(f'Safety Probability vs Lateral and Heading Error ({method_name})')
    elif key[0] == "e" and key[1] == 'v':
        plt.xlabel('Lateral error $e$ [m]')
        plt.ylabel('Longitudinal velocity $v_x$ [m/s]')
        plt.title(f'Safety Probability vs Lateral Error and Velocity ({method_name})')
    
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
    
    plt.show()
    

###############################################################################
if __name__ == '__main__':
    """
    carla: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=5000 &
    """
    args = parse_args()


    try:  
        rl_env = Env(host=args.host,
                     port=args.port, 
                     time_step=0.05, 
                     custom_map_path="./maps/train.xodr",
                     actor_filter='vehicle.audi.tt',
                     spawn_method=map_c_before_corner,
                     spectator_init={'x':-965, 'y':185, 'z':15, 'pitch':-45, 'yaw':120, 'roll':0}, 
                     waypoint_itvl=3.0,
                     spectator_reset=False, 
                     autopilot=False)
        rl_env.reset()
        agent = load_agent(rl_env, args.log_dir, use_lagrangian=args.use_lagrangian, use_gpu=not args.no_gpu)

        x_vehicle = np.array(rl_env.getVehicleState())
        x_road = np.array(rl_env.getRoadState())

        print(f'x_vehicle: {x_vehicle}')
        print(f'x_road: {x_road}')
        
        if args.use_lagrangian:
            lambda_val = agent.get_lambda()
            print(f"Lagrangian multiplier (lambda) value: {lambda_val:.4f}")

    except KeyboardInterrupt:
        print('\nCancelled by user - safety_probability.py.')

    finally:
        if 'rl_env' in locals():
            rl_env.destroy()

    suffix = "_lagrangian" if args.use_lagrangian else ""
    
    output_dir = 'plot/MapC'
    os.makedirs(output_dir, exist_ok=True)
    
    rslu = 40
    horizon = 5
    velocity = 30
    beta_list = np.linspace(-50, 0, rslu)
    yawr_list = np.linspace(30, 90, rslu)
    
    safe_p = np.zeros((len(beta_list), len(yawr_list)))
    
    state_template = np.concatenate([x_vehicle, x_road, np.array([horizon])])
    
    for i in range(len(beta_list)):
        print(f'Processing row {i+1}/{len(beta_list)}')
        
        for j in range(len(yawr_list)):
            state = state_template.copy()
            state[0] = velocity
            state[1] = beta_list[i]
            state[2] = yawr_list[j]
            
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(agent.device)
            
            with torch.no_grad():
                action = agent.actor(state_tensor)
                
                q_val, _ = agent.critic(state_tensor, action)
                safe_p[i][j] = q_val.item()
    
    contour_plot(
        x=beta_list,
        y=yawr_list, 
        z=safe_p,
        use_lagrangian=args.use_lagrangian,
        filename=f'{output_dir}/mapC_td3_cont{suffix}_safe_prob.png'
    )
    
    np.save(f'{output_dir}/mapC_td3_cont{suffix}_safe_prob_data.npy', {
        'beta_list': beta_list,
        'yawr_list': yawr_list,
        'safe_p': safe_p
    })
    
    print(f"{output_dir}/mapC_td3_cont{suffix}_safe_prob.png")
