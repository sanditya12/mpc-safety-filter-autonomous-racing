import casadi as ca
from casadi import sin, cos, pi
import math
import numpy as np
from time import time
from simulate import simulate
from typing import List, Dict

class MPCComponent:
    Q_x = 5
    Q_y = 5
    Q_theta = 0.2
    R_v = 0.1
    R_omega = 0.1
    v_max = 0.6
    v_min = -v_max
    omega_max = pi / 4
    omega_min = -omega_max

    def __init__(self, N=20, sim_time=20, step_horizon=0.2, rob_diameter=1, vision_horizon = 10):
        self.N = N
        self.sim_time = sim_time
        self.step_horizon = step_horizon
        self.rob_diameter = rob_diameter
        self.has_obs_constraint = False
        self.has_track_constraint = False
        self.vision_horizon = vision_horizon

    def DM2Arr(self, dm):
        return np.array(dm.full())
    
    def init_symbolic_vars(self):
        # State Symbolic Variables
        self.x = ca.SX.sym("x")
        self.y = ca.SX.sym("y")
        self.theta = ca.SX.sym("theta")

        self.states = ca.vertcat(self.x, self.y, self.theta)
        self.n_states = self.states.numel()

        # Visible Center Points
        self.center_points = ca.SX.sym

        # Control Symbolic Variables
        self.v = ca.SX.sym("v")
        self.omega = ca.SX.sym("omega")
        self.controls = ca.vertcat(self.v, self.omega)
        self.n_controls = self.controls.numel()

        # Matrix containing all states over all time steps + 1 (since it is initial + predictions)
        self.X = ca.SX.sym("X", self.n_states, self.N + 1)

        # Matrix containing all control actions predictions
        self.U = ca.SX.sym("U", self.n_controls, self.N)

        # Parameter vector containing initial and target states
        self.n_params_with_vision = self.n_states + self.vision_horizon*2
        self.P = ca.SX.sym("P", self.n_states + self.vision_horizon * 2+ self.n_controls)

        # state weights matrix (Q_X, Q_Y, Q_THETA)
        self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_theta)

        # controls weights matrix
        self.R = ca.diagcat(self.R_v, self.R_omega)

    def init_cost_fn_and_g_constraints(self):
        # Basic System Mapper Function
        rhs = ca.vertcat(
            self.v @ cos(self.theta), self.v @ sin(self.theta), self.omega
        )  # right hand side
        self.f = ca.Function("f", [self.states, self.controls], [rhs])


        self.cost_fn = (self.P[self.n_params_with_vision: ] - self.U[:, 0]).T @ (self.P[self.n_params_with_vision: ] - self.U[:, 0])
        self.g = (
            self.X[:, 0] - self.P[: self.n_states]
        )  # first constraint element

        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            st_next = self.X[:, k + 1]
            st_next_euler = st + (self.step_horizon * self.f(st, con))
            self.g = ca.vertcat(self.g, st_next - st_next_euler)

    def init_solver(self):
        init_time = time()
        # Preparing the NLP
        OPT_variables = ca.vertcat(
            self.X.reshape(
                (-1, 1)
            ),  # -1 as param means that casadi will automatically find the number of row/columns
            self.U.reshape((-1, 1)),
        )

        nlp_prob = {
            "f": self.cost_fn,
            "x": OPT_variables,
            "g": self.g,
            "p": self.P,
        }

        opts = {
            "ipopt": {
                "max_iter": 200,
                "print_level": 0,
                "acceptable_tol": 1e-8,
                "acceptable_obj_change_tol": 1e-6,
            },
            "print_time": 0,
        }

        # Initialize solver
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
        print("Time for initializing solver: ", time() - init_time)

    def init_constraint_args(self):
        # Initialze Optimization Variables Constraints Vector
        lbx = ca.DM.zeros(
            (self.n_states * (self.N + 1) + self.n_controls * self.N), 1
        )
        ubx = ca.DM.zeros(
            (self.n_states * (self.N + 1) + self.n_controls * self.N), 1
        )

        # States Bounds
        lbx[
            0 : self.n_states * (self.N + 1) : self.n_states
        ] = -ca.inf  # X lower bound
        lbx[
            1 : self.n_states * (self.N + 1) : self.n_states
        ] = -ca.inf  # Y lower bound
        lbx[
            2 : self.n_states * (self.N + 1) : self.n_states
        ] = -ca.inf  # theta lower bound

        ubx[
            0 : self.n_states * (self.N + 1) : self.n_states
        ] = ca.inf  # X upper bound
        ubx[
            1 : self.n_states * (self.N + 1) : self.n_states
        ] = ca.inf  # Y upper bound
        ubx[
            2 : self.n_states * (self.N + 1) : self.n_states
        ] = ca.inf  # theta upper bound

        # Controls Bounds
        lbx[
            self.n_states * (self.N + 1) :: self.n_controls
        ] = self.v_min  # V lower bound
        lbx[
            self.n_states * (self.N + 1) + 1 :: self.n_controls
        ] = self.omega_min  # Omega lower bound

        ubx[
            self.n_states * (self.N + 1) :: self.n_controls
        ] = self.v_max  # V upper bound
        ubx[
            self.n_states * (self.N + 1) + 1 :: self.n_controls
        ] = self.omega_max  # Omega upper bound

        self.args = {
            "lbg": ca.DM.zeros(
                (self.n_states * (self.N + 1), 1)
            ),  # constraints must equal 0
            "ubg": ca.DM.zeros(
                (self.n_states * (self.N + 1), 1)
            ),  # constraints must equal 0
            "lbx": lbx,
            "ubx": ubx,
        }

    def add_track_constraints(self, max_distance):
        init_time = time()
        self.has_track_constraint = True
        self.center_points = self.P[self.n_states: self.n_params_with_vision]

        for k in range(self.N + 1):
            state_c = self.find_projection_to_center(
                (self.X[0, k], self.X[1, k]), self.center_points
            )
            constraint = (
                max_distance
                - self.rob_diameter / 2
                - ca.sqrt(
                    ((self.X[0, k] - state_c[0]) ** 2)
                    + ((self.X[1, k] - state_c[1]) ** 2)
                )
            )
            self.g = ca.vertcat(self.g, constraint)
        print("Time for adding track constraints: ", time() - init_time)

    def add_track_args(self):
        lbg = ca.DM.zeros(self.N + 1, 1)
        self.args["lbg"] = ca.vertcat(self.args["lbg"], lbg)

        ubg = ca.DM.zeros(self.N + 1, 1)
        ubg[0 : self.N + 1] = ca.inf
        self.args["ubg"] = ca.vertcat(self.args["ubg"], ubg)

    def step(self, state_current: np.ndarray, action_desired: np.ndarray, visible_center_points: np.ndarray):
        self.state_current = ca.DM(state_current)
        self.action_desired = ca.DM(action_desired)
        self.visible_center_points = ca.DM(visible_center_points)

        u = ca.DM.zeros(self.n_controls, self.N)

        self.args["p"] = ca.vertcat(self.state_current, self.visible_center_points, self.action_desired)
        self.args["x0"] = ca.vertcat(
            ca.reshape(self.X0, self.n_states * (self.N + 1), 1),
            ca.reshape(self.u0, self.n_controls * self.N, 1),
        )
        sol = self.solver(
            x0=self.args["x0"],
            lbx=self.args["lbx"],
            ubx=self.args["ubx"],
            lbg=self.args["lbg"],
            ubg=self.args["ubg"],
            p=self.args["p"],
        )

        u_safe = ca.reshape(
            sol["x"][self.n_states * (self.N + 1) :],
            self.n_controls,
            self.N,
        )
        self.X0 = ca.reshape(
            sol["x"][: self.n_states * (self.N + 1)],
            self.n_states,
            self.N + 1,
        )
        self.u0 = ca.horzcat(
            u_safe[:, 1:], ca.reshape(u_safe[:, -1], -1, 1)
        )  # Recycling Previous Controls Predicition
        self.X0 = ca.horzcat(
            self.X0[:, 1:], ca.reshape(self.X0[:, -1], -1, 1)
        )  # Recycling States Matrix

        return u_safe

    def init_sim_params(self):
        self.cat_states = self.DM2Arr(self.X0)
        self.cat_controls = self.DM2Arr(self.u0[:, 0])
        self.times = np.array([[0]])
        self.t0 = 0
        self.t = ca.DM(self.t0)
        self.mpc_iter = 0
    
    def prepare_step(self, state_init: np.ndarray):
        # self.init_symbolic_vars()
        # self.init_cost_fn_and_g_constraints()
        # self.init_solver()
        # self.init_constraint_args()

        self.mpc_completed = False

        self.state_init = ca.DM(state_init)
        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(self.state_init, 1, self.N + 1)

        return

    def step_with_sim_params(self, state_current: np.ndarray, action_desired: np.ndarray, visible_center_points: np.ndarray):
        self.state_current = ca.DM(state_current)
        self.action_desired = ca.DM(action_desired)
        self.visible_center_points = ca.DM(visible_center_points)

        u = ca.DM.zeros(self.n_controls, self.N)
        t1 = time()
        self.args["p"] = ca.vertcat(self.state_current, self.visible_center_points, self.action_desired)
        self.args["x0"] = ca.vertcat(
            ca.reshape(self.X0, self.n_states * (self.N + 1), 1),
            ca.reshape(self.u0, self.n_controls * self.N, 1),
        )
        sol = self.solver(
            x0=self.args["x0"],
            lbx=self.args["lbx"],
            ubx=self.args["ubx"],
            lbg=self.args["lbg"],
            ubg=self.args["ubg"],
            p=self.args["p"],
        )

        u_safe = ca.reshape(
            sol["x"][self.n_states * (self.N + 1) :],
            self.n_controls,
            self.N,
        )
        self.X0 = ca.reshape(
            sol["x"][: self.n_states * (self.N + 1)],
            self.n_states,
            self.N + 1,
        )

        self.cat_states = np.dstack((self.cat_states, self.DM2Arr(self.X0)))
        self.cat_controls = np.vstack(
            (self.cat_controls, self.DM2Arr(u[:, 0]))
        )
        self.t = np.vstack((self.t, self.t0))
        self.t0 += self.step_horizon

        self.u0 = ca.horzcat(
            u_safe[:, 1:], ca.reshape(u_safe[:, -1], -1, 1)
        )  # Recycling Previous Controls Predicition
        self.X0 = ca.horzcat(
            self.X0[:, 1:], ca.reshape(self.X0[:, -1], -1, 1)
        )  # Recycling States Matrix
        t2 = time()
        self.times = np.vstack((self.times, t2 - t1))
        return u_safe


    def get_simulation_params(self):
        return {
            "cat_states": self.cat_states,
            "cat_controls": self.cat_controls,
            "times": self.times,
            "step_horizon": self.step_horizon,
            "N": self.N,
            "p_arr": np.array(
                [
                    self.state_init[0],
                    self.state_init[1],
                    self.state_init[2],
                ]
            ),
            "rob_diam": self.rob_diameter,
        }
    
    def simulate_step_shift(self, u, state_init) -> np.ndarray:
        f_value = self.f(state_init, u)
        return self.DM2Arr( ca.DM(state_init + (self.step_horizon * f_value)))

    # Helper Function for Track Constraint (Finding Projection to Center Line)
    def find_projection_on_segment(self, point:ca.SX, a:ca.SX, b:ca.SX):
        ap = [point[0] - a[0], point[1] - a[1]]
        ab = [b[0] - a[0], b[1] - a[1]]

        ap_ab = ap[0] * ab[0] + ap[1] * ab[1]
        ab2 = ab[0] * ab[0] + ab[1] * ab[1]

        t = ca.if_else(
            ab2 != 0, ap_ab / ab2, 0
        )  # Protect against division by zero
        t = ca.fmin(1, ca.fmax(0, t))  # Constrain t to the interval [0, 1]

        return ca.vertcat(a[0] + ab[0] * t, a[1] + ab[1] * t)

    def find_projection_to_center(self, position: ca.SX, center_points: ca.SX) -> ca.SX:
        num_points = center_points.size1() // 2
        closest_distance = ca.inf
        projected_point = ca.SX([0, 0])

        for i in range(num_points - 1):
            a = [center_points[2*i], center_points[2*i+1]]
            b = [center_points[2*(i + 1)], center_points[2*(i + 1)+1]]

            current_projected_point = self.find_projection_on_segment(
                position, a, b
            )

            distance_squared = (
                position[0] - current_projected_point[0]
            ) ** 2 + (position[1] - current_projected_point[1]) ** 2

            projected_point = ca.if_else(
                distance_squared < closest_distance,
                current_projected_point,
                projected_point,
            )
            closest_distance = ca.if_else(
                distance_squared < closest_distance,
                distance_squared,
                closest_distance,
            )

        return projected_point