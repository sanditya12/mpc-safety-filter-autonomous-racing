from mpc_class import MPCComponent
from simulate import simulate
import numpy as np
from casadi import sin, cos, pi
from track_data import center_sample_points
from time import time
from reference_generator import ReferenceGenerator
import random
import csv


center_points = center_sample_points.center_points

x_init = 1.5
y_init = 5
theta_init = -pi/2
horizon = 8
mpc = MPCComponent(vision_horizon=horizon, N=20)

state_init = np.array([x_init, y_init, theta_init])
mpc.init_symbolic_vars()
mpc.init_cost_fn_and_g_constraints()


start_time = time()
lane_width = 3
max_distance =( lane_width/2)
mpc.add_track_constraints(max_distance)

mpc.init_solver()

mpc.init_constraint_args()

mpc.add_track_args()

mpc.prepare_step(state_init)
mpc.init_sim_params()
cat_real_state = [state_init]

# Open (or create) a CSV file for writing
with open("output.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header
    writer.writerow(["x", "y", "theta", "v_desired","omega_desired","v_safe", "omega_safe" ])
    while time() - start_time < 6:
        init_time = time()

        rg = ReferenceGenerator(horizon, center_points)
        visible_center_points = rg.generate_map((state_init[0], state_init[1]))
        visible_center_points = np.array(visible_center_points).flatten()

        random_v = random.uniform(0, 0.6)
        # random_v = 0.6
        random_omega = random.uniform(-pi, pi)
        # random_omega = 0
        u_desired = np.array([random_v, random_omega])
        u_safe = mpc.step_with_sim_params(
            state_init, u_desired, visible_center_points
        )
        u0_safe = u_safe[:,0]
        writer.writerow([state_init[0],state_init[1],state_init[2], u_desired[0], u_desired[1], u0_safe[0], u0_safe[1]])
        state_init = mpc.simulate_step_shift(u0_safe, state_init)
        cat_real_state.append(state_init)
        # state_arr = np.array(state_init)
        # print("Time per step: ", time() - init_time)
        # # print("x: ", state_arr[0], ", y: ", state_arr[1], ", th: ", state_arr[2])
        # print("  ")





print("Whole MPC time: ", time() - start_time)

sim_params = mpc.get_simulation_params()
simulate(
    sim_params["cat_states"],
    sim_params["cat_controls"],
    sim_params["times"],
    sim_params["step_horizon"],
    sim_params["N"],
    sim_params["p_arr"],
    sim_params["rob_diam"],
    cat_real_state,
    save=True,
)
