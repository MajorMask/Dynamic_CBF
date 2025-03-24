#%%
import os
import torch
from pathlib import Path    
import time
import numpy as np
from tqdm import tqdm
import json

from splat.gsplat_utils import GSplatLoader
from cbf.cbf_utils import CBF
from dynamics.systems import DoubleIntegrator, double_integrator_dynamics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters for the CBF and dynamics #NEED TO UNDERSTAND
alpha = 5.
beta = 1.
dt = 0.05

# Methods for the simulation
n = 100         # number of different configurations
n_steps = 500   # number of time steps #NEED TO UNDERSTAND

# Creates a circle for the configuration
t = np.linspace(0, 2*np.pi, n) # NEED TO TRACK
t_z = 10*np.linspace(0, 2*np.pi, n)

### ----------------- Possible Distance Types ----------------- ###
# method = 'ball-to-ellipsoid'
# method = 'ball-to-ball-squared'
# method = 'ball-to-pt-squared'
# method = 'mahalanobis'
# method = 'ball-to-ball'
### ----------------- Possible Distance Types ----------------- ###

### TODO: SET PIPELINE ###

radius_z = 0.01     # How far to undulate up and down
radius =    0.015       # radius of robot
radius_config = 0.545/2
mean_config = np.array([0.19, 0.01, -0.02])
path_to_gsplat = Path()## TODO: CHANGE INPUT ##
tnow = time.time()
gsplat = GSplatLoader(path_to_gsplat, device)
print('Time to load GSplat:', time.time() - tnow)

dynamics = DoubleIntegrator(device=device, ndim=3)
### Create configurations in a circle
# x0 = np.stack([radius_config*np.cos(t), radius_config*np.sin(t), radius_z * np.sin(t_z)], axis=-1)     # starting positions
# x0 = x0 + mean_config
# xf = np.stack([radius_config*np.cos(t + np.pi), radius_config*np.sin(t + np.pi), radius_z * np.sin(t_z + np.pi)], axis=-1)     # goal positions
# xf = xf + mean_config
    # Run simulation
total_data = []
goal = 0 ## TODO: CHANGE THIS
x = 0 ## TODO: CHANGE THIS
    # for trial, (start, goal) in enumerate(zip(x0, xf)):
    #     # State is 6D. First 3 are position, last 3 are velocity. Set initial and final velocities to 0
    #     x = torch.tensor(start).to(device).to(torch.float32)
    #     x = torch.cat([x, torch.zeros(3).to(device).to(torch.float32)])
    #     goal = torch.tensor(goal).to(device).to(torch.float32)
    #     goal = torch.cat([goal, torch.zeros(3).to(device).to(torch.float32)])
traj = [x] ## TODO: CHANGE THIS
times = [0]
u_values = []
u_des_values = []
safety = []
sucess = []
feasible = []
total_time = []
cbf = CBF(gsplat, dynamics, alpha, beta, radius, distance_type="ball-to-ellipsoid")
print("Simulating trajectory {trial}")
        # Simple PD controller to track to goal
## TODO: SET UP A DIFFERENT LOOP ACCORDING TO PIPELINE
for i in tqdm(range(n_steps), desc=f"Simulating trajectory"):

        vel_des = 5.0*(goal[:3] - x[:3])
        vel_des = torch.clamp(vel_des, -0.1, 0.1)
        # add d term
        vel_des = vel_des + 1.0*(goal[3:] - x[3:])
        # cap between -0.1 and 0.1
        u_des = 1.0*(vel_des - x[3:])
        # cap between -0.1 and 0.1
        u_des = torch.clamp(u_des, -0.1, 0.1)
        ### ----------------- Safety Filtering ----------------- ###
        tnow = time.time()
        torch.cuda.synchronize()
        u = cbf.solve_QP(x, u_des)
        torch.cuda.synchronize()
        ### ----------------- End of Safety Filtering ----------------- ###
        total_time.append(time.time() - tnow)
        
        # We end the trajectory if the solver fails (because we short-circuit the control input if it fails)
        ## TODO: Add error handling here
        if cbf.solver_success == False:
            print("Solver failed")
            sucess.append(False)
            feasible.append(False)
            break
        # Propagate dynamics
        x_ = x.clone()
        x = double_integrator_dynamics(x,u)*dt + x
        traj.append(x)
        times.append((i+1) * dt)
        u_values.append(u.cpu().numpy())
        u_des_values.append(u_des.cpu().numpy())
        h = cbf.query_distance(x[:3], radius=radius) ## TODO: CHANGE THIS
        safety.append(torch.min(h).item())
        # It's not moving
        ## TODO: SET UP A DIFFERENT LOOP ACCORDING TO PIPELINE
        if torch.norm(x - x_) < 0.001:
            # If it's at the goal
            # if torch.norm(x_ - goal) < 0.001:
            #     print("Reached Goal")
            #     sucess.append(True)
            #     feasible.append(True)
            # else:
            #     sucess.append(False)
            #     feasible.append(True)
            break
        # If it times out but still moving, we consider this loosely a success. 
        if i >= n_steps - 1:
            sucess.append(True)
            feasible.append(True)
traj = torch.stack(traj)
u_values = np.array(u_values)
u_des_values = np.array(u_des_values)
data = {
'traj': traj.cpu().numpy().tolist(),
'u_out': u_values.tolist(),
'u_des': u_des_values.tolist(),
'time_step': times,
'safety': safety,
'sucess': sucess,
'feasible': feasible,
'total_time': total_time,
'cbf_solve_time': cbf.times_cbf,
'qp_solve_time': cbf.times_qp,
'prune_time': cbf.times_prune,
}
total_data.append(data)
# Save trajectory
data = {
    'alpha': alpha,
    'beta': beta,
    'radius': radius,
    'dt': dt,
    'total_data': total_data,
}
# create directory if it doesn't exist
os.makedirs('trajs', exist_ok=True)
# write to the file
with open(f'trajs/traj.json', 'w') as f:
    json.dump(data, f, indent=4)
# 