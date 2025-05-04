# -*- coding: utf-8 -*-
"""
Verification script for TD3-PIRL normal cornering in Town2
"""
####################################
# general packages
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import torch

#######################################
# TD3-PIRL agent and CarEnv
sys.path.append(os.pardir)
sys.path.append('.')

from rl_env.carla_env import CarEnv
from rl_agent.TD3_PIRL_torch import TD3PIRLagent, agentOptions, pinnOptions
from training_td3_pirl_Town2 import convection_model, diffusion_model, sample_for_pinn

# carla environment
class Env(CarEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def reset(self):
        carla_state = super().reset()
        print(carla_state[0:3])
        horizon = 6.0  # always reset with 6.0 (randomized in training)
        self.state = np.array(list(carla_state) + [horizon])        
        return self.state

    def step(self, action):
        """
        Modified to handle continuous actions from TD3
        """
        # Extract steering and throttle from continuous action
        steering = action[0]  # Should be in range [-1, 1]
        throttle = action[1]  # Should be in range [0, 1]
        
        # Map to nearest discrete action index for compatibility with original env
        steering_idx = int(np.round((steering + 1) * (len(self.step_S_pool) - 1) / 2))
        throttle_idx = int(np.round(throttle * (len(self.step_T_pool) - 1)))
        
        # Ensure indices are within valid ranges
        steering_idx = max(0, min(steering_idx, len(self.step_S_pool) - 1))
        throttle_idx = max(0, min(throttle_idx, len(self.step_T_pool) - 1))
        
        # Calculate action index
        action_idx = throttle_idx * len(self.step_S_pool) + steering_idx
        
        # Make a step
        new_veh_state, reward, done = super().step(action_idx)
        horizon = self.state[-1] - self.time_step
        new_state = np.array(list(new_veh_state) + [horizon])
        self.state = new_state

        return new_state, reward, done

###########################################
# Simulation function
###########################################
def closed_loop_simulation(agent, env, T):
    
    # initialization
    current_state = env.reset()    
    state_trajectory = np.zeros([len(current_state), int(T/env.time_step)])
    vehicle_trajectory = np.zeros([3, int(T/env.time_step)])  # (x,y,yaw)
    waypoints = []
    
    for i in range(int(T/env.time_step)):
        # Get continuous action from TD3 actor
        state_tensor = torch.FloatTensor(current_state.reshape(1, -1))
        with torch.no_grad():
            action = agent.actor(state_tensor).cpu().numpy().flatten()
        
        new_state, _, is_done = env.step(action)
        
        # get waypoint from new_state
        vehicle_locat = env.vehicle.get_transform().location
        way_point = env.world.get_map().get_waypoint(vehicle_locat, project_to_road=True)
        
        right_way_point = way_point.get_right_lane()
        left_way_point = way_point.get_left_lane()
        way_point = right_way_point if right_way_point.lane_width > left_way_point.lane_width else left_way_point

        _, wps = env.fetch_relative_states(way_point.transform, 0.5, 5)
        
        waypoints.append(wps)
        
        current_state = new_state   
        state_trajectory[:,i] = new_state
        
        # position
        vehicle_trajectory[:,i] = env.get_vehicle_position()  
        
    return state_trajectory, vehicle_trajectory, waypoints


################################################################################
# Main
################################################################################
if __name__ == '__main__':
    """
    run carla by: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=3000 &
    """

    ####################################
    # Settings
    ####################################
    data_dir = 'logs/Town2/TD3_04070458'  # Update with your TD3 trained model path
    check_point = 10_000

    carla_port = 3000
    time_step = 0.05 
    video_save = None #'plot/Town2/TD3_simulation.mp4'
    spec_town2 = {'x':-7.39, 'y':312, 'z':10.2, 'pitch':-20, 'yaw':-45, 'roll':0}  # spectator coordinate  

    #################################
    # Environment
    #################################
    def choose_spawn_point(carla_env):
        sp_list = carla_env.get_all_spawn_points()    
        spawn_point = sp_list[1]
        return spawn_point
    
    def vehicle_reset_method(): 
        x_loc = 0
        y_loc = 0  # np.random.uniform(-0.5,0.5) 
        psi_loc = 0  # np.random.uniform(-20,20)
        vx = np.random.uniform(10, 10)
        vy = 0 
        yaw_rate = 0   
        return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]
    
    env = Env(port=carla_port, 
              time_step=time_step,
              custom_map_path=None,
              actor_filter='vehicle.audi.tt',  
              spawn_method=choose_spawn_point,
              vehicle_reset=vehicle_reset_method,
              waypoint_itvl=0.5,
              spectator_init=spec_town2,  
              spectator_reset=False,
              camera_save=video_save,
              )
    
    state_dim = len(env.reset())
    action_dim = 2  # [steering, throttle]
    max_action = np.array([1.0, 1.0])  # Max values for steering and throttle

    #################################################
    # Load TD3-PIRL agent
    #################################################
    agentOp = agentOptions()
    pinnOp = pinnOptions(convection_model, diffusion_model, sample_for_pinn)
    
    # Create and load agent
    agent = TD3PIRLagent(state_dim, action_dim, max_action, agentOp, pinnOp)
    agent.load_weights(data_dir, ckpt_idx=check_point)

    ######################################
    # Closed loop simulation
    ######################################
    random.seed(1)
    np.random.seed(1)
    
    slip_angles_all = []
    yaw_rates_all = []
    
    for i in range(5):
        print(f"==================Vehicle {i+1}==================")
        T = 5
        states, positions, waypoints = closed_loop_simulation(agent, env, T)
        
        # Save trajectories if needed
        # np.savez(f'plot/Town2/TD3_data{i}', state=states, position=positions)
        
        slip_angle, yaw_rate = states[1, :], states[2, :]
        slip_angles_all.append(slip_angle)
        yaw_rates_all.append(yaw_rate)
        
        # Plot vehicle trajectory
        x = positions[0, :]
        y = positions[1, :]
        plt.plot(y, x, color='blue', alpha=0.5, lw=0.5, label=f"vehicle {i+1}")
        plt.scatter(y[0], x[0], color='blue', marker='x')

    if 'env' in locals():
        env.destroy()

    #####################################
    # Plot slip angle and yaw rate
    #####################################
    plt.title("Vehicle Trajectories with TD3-PIRL Control (Town2)")
    plt.xlabel("Y Position")
    plt.ylabel("X Position")
    plt.savefig("plot/Town2/TD3_trajectories.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    time_steps = np.arange(len(slip_angles_all[0])) * env.time_step
    
    # Plot slip angle
    plt.figure(figsize=(10, 5))
    for i, s in enumerate(slip_angles_all):
        plt.plot(time_steps, s, label=f'Vehicle {i+1}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Slip Angle (degrees)')
    plt.title('Slip Angle vs Time (TD3-PIRL in Town2)')
    plt.legend()
    plt.grid(True)
    plt.savefig("plot/Town2/TD3_slip_angles.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Plot yaw rate
    plt.figure(figsize=(10, 5))
    for i, y in enumerate(yaw_rates_all):
        plt.plot(time_steps, y, label=f'Yaw Rate {i+1}')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Yaw Rate (degrees/second)')
    plt.title('Yaw Rate vs Time (TD3-PIRL in Town2)')
    plt.legend()
    plt.grid(True)
    plt.savefig("plot/Town2/TD3_yaw_rates.png", dpi=300, bbox_inches="tight")
    plt.show()
