# td3-lagrangian-pirl
*28th IEEE International Conference on Intelligent Transportation Systems (ITSC 2025)*

## Requirements
1. Tested on Ubuntu 22.04.4 LTS
2. Nvidia GPU equipped, and driver Installed. Tested on GeForce RTX 3080/RTX 5070 Ti. 
3. Install [CARLA simulator](https://carla.org/), which is an open-source simulator for autonomous driving research. Tested on CARLA 0.9.15. 
4. Install [Anaconda](https://www.anaconda.com/), which is a package manager, environment manager, and Python distribution.
5. Setup conda environment
   ```console
    conda env create -f environment.yml
   ```
## Directory Structure
td3-lagrangian-pirl

1. pirl_carla: TD3-PIRL safe-drifting on MapC

2. pirl_carla_la: TD3-Lagrangian-PIRL safe-drifting 

3. pirl_carla_lacon: TD3-Lagrangian-PIRL safe-drifting (full continuous-action)

4. pirl_carla_town2: TD3-PIRL high-speed turning on Town2

## Quick Start


### 1. Setup environment and CARLA
```
conda activate td3_pirl
```
### 2. Safe Drifting (TD3-PIRL,TD3-Lagrangian-PIRL(con))
```
cd pirl_carla
~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=4000 &
python training_td3_pirl_MapC.py
```
```
~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=5000 &
python verification_TD3_MapC.py
python safety_probability_td3_mapC.py
```
### 3. High-Speed Turning (Town2)
```
cd pirl_carla_town2
~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=2000 &
python training_td3_pirl_Town2.py
```
```
~/carla/carla_0_9_15/CarlaUE4.sh -carla-rpc-port=3000 &
python verification_TD3_Town2.py
python safety_probability_td3_town2.py
```
> **Note:** When running parts of these scripts, you can set optional parameters (e.g., `--no_gpu`, `--lr`, etc.).
