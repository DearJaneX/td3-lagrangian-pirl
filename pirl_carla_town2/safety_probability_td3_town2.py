"""
Plot safe probability vs map for TD3-PIRL in Town2
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# TD3-PIRL agent and CarEnv
sys.path.append(os.pardir)
sys.path.append('.')

from rl_agent.TD3_PIRL_torch import TD3PIRLagent, agentOptions, pinnOptions
from training_td3_pirl_Town2 import Env, convection_model, diffusion_model, sample_for_pinn

###########################################################################
# Settings

carla_port = 3000
time_step = 0.05    
spec_town2 = {'x':-7.39, 'y':312, 'z':10.2, 'pitch':-20, 'yaw':-45, 'roll':0}    
key = ["e", "psi"]  # v or psi 

# Update with your trained TD3 model path
log_dir = 'logs/Town2/TD3_04070458'
check_point = 10_000

###########################################################################
# Load TD3-PIRL agent and carla environment     
def load_agent(env, log_dir):
    state_dim = len(env.reset())
    action_dim = 2  # [steering, throttle]
    max_action = np.array([1.0, 1.0])  # Max values for steering and throttle
    
    agentOp = agentOptions()
    pinnOp = pinnOptions(convection_model, diffusion_model, sample_for_pinn)
    agent = TD3PIRLagent(state_dim, action_dim, max_action, agentOp, pinnOp)
    agent.load_weights(log_dir, ckpt_idx=check_point)
    
    return agent


def contour_plot(x, y, z, key=["e", "psi"], filename=None):
    # (x, y) = (e, psi)/ (e, vx)
    x, y = np.meshgrid(x, y)
    x = x.transpose()
    y = y.transpose()
    
    # Creating the contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x, y, z, cmap='viridis')
    cbar = plt.colorbar(contour)  # Adding color bar to show z values    
    cbar.set_label('Safety Probability')
    plt.rcParams.update({'font.size': 15})    
    
    # Adding labels and title
    if key[0] == "e" and key[1] == 'psi':
        plt.xlabel('Lateral error $e$ [m]')
        plt.ylabel(r'Heading error $\psi$ [rad]')
        plt.title('Safety Probability vs Lateral and Heading Error (TD3-PIRL)')
    elif key[0] == "e" and key[1] == 'v':
        plt.xlabel('Lateral error $e$ [m]')
        plt.ylabel('Longitudinal velocity $v_x$ [m/s]')
        plt.title('Safety Probability vs Lateral Error and Velocity (TD3-PIRL)')
    
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=300)

    # Displaying the plot
    plt.show()

    
###############################################################################
if __name__ == '__main__':

    try:  
        ###################################
        # Load env and agent
        ##################################
        rl_env = Env(port=carla_port, 
                      time_step=time_step, 
                      custom_map_path=None,
                      spawn_method=None, 
                      spectator_init=spec_town2, 
                      spectator_reset=False, 
                      autopilot=False)
        
        agent = load_agent(rl_env, log_dir)

        ##############################
        # Define analysis grid parameters   
        rslu = 50
        psi_scale = 0.4
        e_scale = 1.0
        e_list = np.linspace(-e_scale, e_scale, rslu)
        v_list = np.linspace(5, 25, rslu)
        psi_list = np.linspace(-psi_scale, psi_scale, rslu)
        
        #############################
        # Get waypoints
        #############################
        interval = 0.5
        next_num = 0
        
        if key[1] == "psi":
            eps = 0.1  # initial distance
        else:
            eps = 0.5  # initial distance
        
        refer_point = rl_env.test_random_spawn_point(eps=eps)
        vector, waypoints = rl_env.fetch_relative_states(wp_transform=refer_point, 
                                                        interval=interval, 
                                                        next_num=next_num)
        print(vector)
        
        ##############################
        # Calculate safety probability using TD3 critic
        #############################
        x_vehicle = np.array([10, 0, 0])
        horizon = 5

        # Initialize the safety probability grid
        safe_p = np.zeros((len(e_list), len(psi_list)))
        
        # Determine which variable to use for y-axis
        if key[1] == "v":
            y_list = v_list
        else:
            y_list = psi_list
        
        # Create a basic state structure
        x_road = [0, 0]
        
        # Evaluate the safety probability for each grid point
        for i in range(len(e_list)):
            print(f'Processing row {i+1}/{len(e_list)}')
            x_road[0] = e_list[i]
            
            for j in range(len(y_list)):
                if key[1] == "v":
                    x_vehicle[0] = y_list[j]
                else:
                    x_road[1] = y_list[j]
                
                # Construct the full state
                new_x_road = x_road + vector
                state = np.concatenate([x_vehicle, new_x_road, np.array([horizon])])
                
                # Convert to tensor for neural network input
                state_tensor = torch.FloatTensor(state.reshape(1, -1))
                
                # Use the TD3 actor and critic to evaluate safety probability
                with torch.no_grad():
                    # Get action from actor
                    action = agent.actor(state_tensor)
                    
                    # Use critic to evaluate safety probability
                    q_val, _ = agent.critic(state_tensor, action)
                    safe_p[i][j] = q_val.item()
        
        #################################       
        # Save and plot results
        ##################################
        # Save data for future reference
        np.savez(f"plot/Town2/TD3_safe_prob_{key[0]}_{key[1]}.npz", 
                 x=e_list, y=y_list, z=safe_p)
        
        # Generate and save plot
        contour_plot(x=e_list, y=y_list, z=safe_p, key=key,
                     filename=f'plot/Town2/TD3_safe_prob_{key[0]}_{key[1]}.png')

    except KeyboardInterrupt:
        print('\nCancelled by user - safety_probability.py.')

    finally:
        if 'rl_env' in locals():
            rl_env.destroy()
