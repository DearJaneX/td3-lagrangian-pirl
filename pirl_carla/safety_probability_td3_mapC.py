"""
Plot safe probability vs map for TD3-PIRL implementation
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.pardir)
sys.path.append('.')
from rl_agent.TD3_PIRL_torch import TD3PIRLagent, agentOptions, pinnOptions
from training_td3_pirl_MapC import Env, convection_model, diffusion_model, sample_for_pinn, map_c_before_corner

###########################################################
# Settings
###########################################################
log_dir = "logs/MapC/TD3_04051838"

carla_port = 5000
time_step = 0.05
map_train = "./maps/train.xodr"
spec_mapC_NorthEast = {'x':-965, 'y':185, 'z':15, 'pitch':-45, 'yaw':120, 'roll':0} 

#############################################################
# Load TD3-PIRL agent and carla environment     
############################################################

def load_agent(env, log_dir):
    state_dim = len(env.reset())
    action_dim = 2  # [steering, throttle]
    max_action = np.array([1.0, 1.0])  # Max values for steering and throttle
    
    agentOp = agentOptions()
    pinnOp = pinnOptions(convection_model, diffusion_model, sample_for_pinn)    
    agent = TD3PIRLagent(state_dim, action_dim, max_action, agentOp, pinnOp)
    agent.load_weights(log_dir, ckpt_idx='latest')

    return agent


def contour_plot(x, y, z, key=None, filename=None):
    #(x, y) = (e, psi)/ (e, vx)
    x, y = np.meshgrid(x, y)
    x = x.transpose()
    y = y.transpose()
    
    # Creating the contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x, y, z, cmap='viridis')
    cbar = plt.colorbar(contour)  # Adding color bar to show z values
    cbar.set_label('Safety Probability')
    
    # Adding labels and title
    if not key:
        plt.xlabel(r'Slip angle $\beta$ [deg]')
        plt.ylabel('Yaw rate $r$ [deg/s]')
        plt.title('Safety Probability Visualization (TD3-PIRL)')
    elif key[0] == "e" and key[1] == 'psi':
        plt.xlabel('Lateral error $e$ [m]')
        plt.ylabel(r'Heading error $\psi$ [rad]')
        plt.title('Safety Probability vs Lateral and Heading Error (TD3-PIRL)')
    elif key[0] == "e" and key[1] == 'v':
        plt.xlabel('Lateral error $e$ [m]')
        plt.ylabel('Longitudinal velocity $v_x$ [m/s]')
        plt.title('Safety Probability vs Lateral Error and Velocity (TD3-PIRL)')
    
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
    
    plt.show()
    

###############################################################################
if __name__ == '__main__':
    """
    run carla by: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=5000 &
    """    

    ###################################
    # Load env and agent
    ##################################
    try:  
        # Get reference state
        rl_env = Env(port=carla_port, 
                     time_step=time_step, 
                     custom_map_path=map_train,
                     actor_filter='vehicle.audi.tt',
                     spawn_method=map_c_before_corner,
                     spectator_init=spec_mapC_NorthEast, 
                     waypoint_itvl=3.0,
                     spectator_reset=False, 
                     autopilot=False)
        rl_env.reset()
        agent = load_agent(rl_env, log_dir)

        x_vehicle = np.array(rl_env.getVehicleState())
        x_road = np.array(rl_env.getRoadState())

        print(f'x_vehicle: {x_vehicle}')
        print(f'x_road: {x_road}')

    except KeyboardInterrupt:
        print('\nCancelled by user - safety_probability.py.')

    finally:
        if 'rl_env' in locals():
            rl_env.destroy()

    #####################################
    # Safety probability  
    #####################################
    
    rslu = 40
    horizon = 5
    velocity = 30
    beta_list = np.linspace(-50, 0, rslu)
    yawr_list = np.linspace(30, 90, rslu)
    
    safe_p = np.zeros((len(beta_list), len(yawr_list)))
    
    # Create a basic state template
    state_template = np.concatenate([x_vehicle, x_road, np.array([horizon])])
    
    for i in range(len(beta_list)):
        print(f'Processing row {i+1}/{len(beta_list)}')
        
        for j in range(len(yawr_list)):
            # Create modified state with current beta and yaw rate values
            state = state_template.copy()
            state[0] = velocity
            state[1] = beta_list[i]
            state[2] = yawr_list[j]
            
            # Get the safety probability using TD3's critic network
            state_tensor = torch.FloatTensor(state.reshape(1, -1))
            
            with torch.no_grad():
                # Get action from actor
                action = agent.actor(state_tensor)
                
                # Use critic to evaluate safety probability
                q_val, _ = agent.critic(state_tensor, action)
                safe_p[i][j] = q_val.item()
    
    # Plot results
    contour_plot(
        x=beta_list,
        y=yawr_list, 
        z=safe_p,
        filename='plot/MapC/mapC_td3_safe_prob.png'
    )
    
    # Also save the data for future reference
    np.save('plot/MapC/mapC_td3_safe_prob_data.npy', {
        'beta_list': beta_list,
        'yawr_list': yawr_list,
        'safe_p': safe_p
    })
