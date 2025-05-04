
####################################
# general packages
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import argparse
import os
import sys

#######################################
# TD3-PIRL agent and CarEnv
#######################################
from rl_agent.TD3_PIRL_torch import TD3PIRLagent, agentOptions, pinnOptions
from rl_env.continuous_carla_env import ContinuousCarEnv
from rl_env.carla_env import map_c_before_corner, road_info_map_c_north_east
from training_td3_pirl_MapC import convection_model, diffusion_model, sample_for_pinn

class Env(ContinuousCarEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def reset(self):
        carla_state = super().reset()
        print(carla_state[0:3])
        horizon = 5.0
        self.state = np.array(list(carla_state) + [horizon])        
        return self.state

    def step(self, action):

        new_veh_state, reward, done = super().step(action)
        horizon = self.state[-1] - self.time_step
        new_state = np.array(list(new_veh_state) + [horizon])
        self.state = new_state

        return new_state, reward, done


def closed_loop_simulation(agent, env, T):
    
    current_state = env.reset()    
    state_trajectory = np.zeros([len(current_state), int(T/env.time_step)])
    vehicle_trajectory = np.zeros([3, int(T/env.time_step)])  # (x,y,yaw)
    waypoints = []
    actions_taken = []  
    
    for i in range(int(T/env.time_step)):
        state_tensor = torch.FloatTensor(current_state.reshape(1, -1)).to(agent.device)
        with torch.no_grad():
            action = agent.actor(state_tensor).cpu().numpy().flatten()
        
        actions_taken.append(action.copy())
        
        new_state, _, is_done = env.step(action)
        
        vehicle_locat = env.vehicle.get_transform().location
        way_point = env.world.get_map().get_waypoint(vehicle_locat, project_to_road=True)
        
        right_way_point = way_point.get_right_lane()
        left_way_point = way_point.get_left_lane()
        way_point = right_way_point if right_way_point.lane_width > left_way_point.lane_width else left_way_point

        _, wps = env.fetch_relative_states(way_point.transform, 0.5, 5)  # 0.5, 5
        
        waypoints.append(wps)
        
        current_state = new_state   
        state_trajectory[:,i] = new_state
        
        vehicle_trajectory[:,i] = env.get_vehicle_position()  
        
    return state_trajectory, vehicle_trajectory, waypoints, np.array(actions_taken)



if __name__ == '__main__':
    """
    carla: 
        ~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=5000 &
    """    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="logs/MapC/TD3_Cont_Lagrangian_04181330")
    parser.add_argument('--check_point', type=int, default=19000)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default="localhost")
    parser.add_argument('--use_lagrangian', action='store_true')
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()
    

    data_dir = args.data_dir
    check_point = args.check_point
    
    log_dir = 'plot/MapC/data_trained_td3_cont_lagrangian' if args.use_lagrangian else 'plot/MapC/data_trained_td3_cont'
    os.makedirs(log_dir, exist_ok=True)
    
    video_save = None  # 'plot/MapC/simulation_td3_continuous.mp4'

    carla_port = args.port
    carla_host = args.host
    time_step = 0.05 
    map_train = "./maps/train.xodr"
    spectator_view = {'x':-965, 'y':185, 'z':15, 'pitch':-45, 'yaw':120, 'roll':0} 


    def vehicle_reset_method_():
        x_loc = 0
        y_loc = 0 
        psi_loc = 0  
        vx = 30
        vy = -vx*np.random.uniform(np.tan(20/180*3.14), np.tan(25/180*3.14))
        yaw_rate = np.random.uniform(60, 70)
        
        return [x_loc, y_loc, psi_loc, vx, vy, yaw_rate]  
    
    env = Env(host=carla_host,
              port=carla_port, 
              time_step=time_step,
              custom_map_path=map_train,
              actor_filter='vehicle.audi.tt',  
              spawn_method=map_c_before_corner, 
              vehicle_reset=vehicle_reset_method_,
              waypoint_itvl=3.0,
              spectator_init=spectator_view, 
              spectator_reset=False, 
              camera_save=video_save,
              camera_view=spectator_view,
              camera_fov=90,
              )
    
    state_dim = len(env.reset())
    action_dim = 2  # [steering, throttle]
    max_action = np.array([1.0, 1.0]) 

    sp, left_pt, right_pt = road_info_map_c_north_east(env, 100)
    suffix = '_lagrangian' if args.use_lagrangian else ''
    np.savez(log_dir+'/../spawn_points_td3_cont'+suffix, 
             center=np.array(sp), left=np.array(left_pt), right=np.array(right_pt))


    agentOp = agentOptions(USE_LAGRANGIAN=args.use_lagrangian, USE_GPU=not args.no_gpu)
    pinnOp = pinnOptions(convection_model, diffusion_model, sample_for_pinn)
    
    agent = TD3PIRLagent(state_dim, action_dim, max_action, agentOp, pinnOp)
    agent.load_weights(data_dir, ckpt_idx=check_point) 

    if args.use_lagrangian:
        lambda_val = agent.get_lambda()
        print(f"Lagrangian multiplier (lambda) value: {lambda_val:.4f}")

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    slip_angles_all = []
    yaw_rates_all = []
    steering_actions = []
    throttle_actions = []
    
    for i in range(20):
        print(f"==================Vehicle {i+1}==================")    
    
        T = 5

        states, positions, waypoints, actions = closed_loop_simulation(agent, env, T)
  
        slip_angle, yaw_rate = states[1, :], states[2, :]
        slip_angles_all.append(slip_angle)
        yaw_rates_all.append(yaw_rate)
        
        steering_actions.append(actions[:, 0])
        throttle_actions.append(actions[:, 1])

        x = positions[0,:]
        y = positions[1,:]
        plt.plot(y, x, color='blue', alpha=0.5, lw=0.5, label=f"vehicle {i+1}")
        plt.scatter(y[0], x[0], color='blue', marker='x')

    if 'env' in locals():
        env.destroy()


    method_name = "TD3-PIRL with Continuous Actions" + (" and Lagrangian" if args.use_lagrangian else "")
    plt.title(f"Vehicle Trajectories with {method_name}")
    plt.xlabel("Y Position")
    plt.ylabel("X Position")
    plt.savefig(f"{log_dir}/trajectories_td3_cont{suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    time_steps = np.arange(len(slip_angles_all[0])) * env.time_step
    
    plt.figure(figsize=(10, 5))
    for i, s in enumerate(slip_angles_all):
        plt.plot(time_steps, s, label=f'Vehicle {i+1}' if i < 3 else None)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Slip Angle (degrees)')
    plt.title(f'Slip Angle Evolution with {method_name}')
    if len(slip_angles_all) <= 5:
        plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_dir}/slip_angles_td3_cont{suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    for i, y in enumerate(yaw_rates_all):
        plt.plot(time_steps, y, label=f'Vehicle {i+1}' if i < 3 else None)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Yaw Rate (degrees/second)')
    plt.title(f'Yaw Rate Evolution with {method_name}')
    if len(yaw_rates_all) <= 5:
        plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_dir}/yaw_rates_td3_cont{suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    for i, steer in enumerate(steering_actions):
        plt.plot(time_steps, steer, label=f'Vehicle {i+1}' if i < 3 else None)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Steering Action [-0.8 to 0.8]')
    plt.title(f'Steering Actions with {method_name}')
    if len(steering_actions) <= 5:
        plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_dir}/steering_actions_td3_cont{suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    for i, throttle in enumerate(throttle_actions):
        plt.plot(time_steps, throttle, label=f'Vehicle {i+1}' if i < 3 else None)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Throttle Action [0.6 to 1.0]')
    plt.title(f'Throttle Actions with {method_name}')
    if len(throttle_actions) <= 5:
        plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_dir}/throttle_actions_td3_cont{suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nSimulation completed for {method_name}.")
    print(f"Results saved to: {log_dir}")
