import gym
from gym import spaces
import numpy as np
import math
import matplotlib.pyplot as plt
import control as ct
import random

class Quadcopter_Pitch_Random (gym.Env):

    def __init__(self):
        super().__init__() #import gym.env init
        print ("This is quadcopter pitch only, random initial pitch mode")
        #####################
        self.MAX_EPISODE = 1000 #1 episode = 1000 steps max
        #threshold untuk masing2 state
        #state ada 3: pitch, pitch dot, wy
        # self.roll_threshold = self.roll_dot_threshold = math.pi
        self.pitch_threshold = self.pitch_dot_threshold = math.pi
        # self.wx_threshold = math.pi
        self.wy_threshold = math.pi
        high = np.array([self.pitch_threshold, self.pitch_dot_threshold, self.wy_threshold],
                        dtype=np.float32)
        self.min_action = np.array([0.1, 0.1])
        self.max_action = np.array([20, 20]) #batas kp kd
        self.action_space = spaces.Box(self.min_action, self.max_action, shape = (2,), dtype = np.float32) #kp kd pitch 

        self.observation_space = spaces.Box(-high, high, shape = (3,), dtype = np.float32) #roll pitch, turunannya, omega x y
        
        ##########################
        self.steps_left = self.MAX_EPISODE

        #state array: pitch, pitch dot, wy -> used for RL training

        _, self.initial_pitch = self._generate_initial_value()
        self.state = [self.initial_pitch, 0, 0] 
        # print("initial pitch init:", self.state[0]) #debug
        #############################

        #Defining constants and parameters
        self.m = 1.776 #quadcopter mass
        self.g = 9.8 #gravity
        self.b = 0.0087 #lift constant
        self.k = 0.0055e-2 #drag constant
        self.d = 0.225 #rotor axis to center of mass distance
        self.B = 0.5 #aerodynamic friction
        self.delta_t = 0.01

        #Intertial properties
        self.Ixx = 0.0035
        self.Iyy = 0.0035
        self.Izz = 0.0055

        self.Ixy = self.Iyx = 0
        self.Ixz = self.Izx = 0
        self.Iyz = self.Izy = 0
        self.J = self.calculate_inertia_matrix()
        self.J_inv = np.linalg.inv(self.J)


        #state feedback definition
        self.A_mat = np.array([[0, 1],
                          [0, (-self.B/self.m)]])
        self.B_mat = np.array ([[0], [1]])
        self.C_mat = np.array ([[1, 0]]) #asumsikan output adalah posisi z
        self.D_mat = np.array ([[0]])
        # self.u = np.empty(0) #to store state feedback input
        
        ###Desired poles
        self.desired_poles = np.array([[complex(-1, 1), complex(-1, -1)]])
        # print(np.shape(desired_poles))

        #### Desired z position (state feedback reference tracking)
        self.r = -1.5

        ##### Desired PD controller output
        # self.desired_roll = 0
        self.desired_pitch = 0
        # self.desired_yaw = 0
        # self.desired_roll_dot = 0
        self.desired_pitch_dot = 0
        # self.desired_yaw_dot = 0

        #### state feedback
        self.K, self.Nx, self.Nu = self.compute_state_feedback(self.A_mat, self.B_mat, self.C_mat, self.D_mat, self.desired_poles)

        ### List initialization
        self.pos_values, self.v_values, self.w_values, self.rpy_values, self.rpy_dot_values = ([] for _ in range(5))
        self.u_values = []
        self.tau_values = []
        self.state_feedback_error_values = []
        self.pd_controller_error_values = []
        self.pd_controller_dot_error_values = []

################## 
    def step(self, action): #main loop
        action = np.clip(action, self.min_action, self.max_action)
        # print("Action:", action)
        Kp_pitch, Kd_pitch = action
        self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw, self.roll_dot, self.pitch_dot, self.yaw_dot, self.wx, self.wy, self.wz = self.full_state
        # print("State:", self.state)

        #Step 1: Hitung thrust masing-masing rotor
        # Ti = [b*(wi**2) for wi in w_bar] = u
        x_sf = np.array([[self.z], [self.vz]]) #Initial condition, z dan vz
        self.u = -(self.K @ x_sf) + (self.Nu + (self.K @ self.Nx))*self.r
        self.u = (self.g - self.u)*self.m
        # print ("u = ", self.u.flatten())

        #Step 2: Hitung torque roll, pitch, yaw axis
        # when T0 = T1 = T2 = T3 = 1/4*T,
        # since u = T = b*w_bar_sq, therefore:
        # w_bar_sq = u[0][0]/b #total w bar sq
        # w_bar_0_sq = w_bar_1_sq = w_bar_2_sq = w_bar_3_sq = (1/4)*w_bar_sq
        # tau_x = d*b*((w_bar_3_sq)-(w_bar_1_sq)) + random.uniform(-0.005, 0.005)
        # tau_y = d*b*((w_bar_0_sq)-(w_bar_2_sq)) #+ random.uniform(-0.005, 0.005)
        # tau_z = k*(w_bar_0_sq - w_bar_1_sq + w_bar_2_sq - w_bar_3_sq) #+ random.uniform(-0.005, 0.005)

        #case: tau pake Kp Kd
        T = self.u[0][0] #state feedback
        # self.tau_x = self.pd_controller(self.desired_roll, self.roll, self.desired_roll_dot, self.roll_dot, Kp_roll, Kd_roll)
        self.tau_x = 0
        self.tau_y = self.pd_controller(self.desired_pitch, self.pitch, self.desired_pitch_dot, self.pitch_dot, Kp_pitch, Kd_pitch)
        self.tau_z = 0 #PD controller not needed
        self.tau = np.array([[self.tau_x], [self.tau_y], [self.tau_z]]) ##?
        thrust_torque_vector = np.vstack(([T], self.tau))
        #cari w_bar_sq masing2 rotor dari invers matriks
        #trus buat apa ya? hanya untuk tau input signalnya saja?
        w_bar_sq_vector = self.get_w_bar_sq(thrust_torque_vector)

        #Step 3: Dinamika rotasi -> hitung omega dot
        w_cross_Jw = np.cross(1*self.w.flatten(), (self.J@self.w).flatten()).reshape(-1,1)
        w_dot = self.J_inv @ (-w_cross_Jw + self.tau)
        # print (w_dot)
        # print("omega:", w_dot)

        # print ("rpy:", rpy)
        # print ("roll", roll)
        # print ("pitch", pitch)
        # print ("yaw", yaw)


        #Step 4: Dinamika translasi -> hitung v dot
        gravity_vector = np.array([[0], [0], [self.m*self.g]])
        R = self.rpy_to_rot(self.roll, self.pitch, self.yaw)
        # T = sum(Ti) -> ga kepake
        # T = u[0][0] #state feedback
        thrust_body = np.array([[0], [0], [T]])
        thrust_inertial = R@thrust_body
        # print(thrust_inertial)
        drag_force = self.B*self.v
        v_dot = (1/self.m)*(gravity_vector - thrust_inertial - drag_force)


        #Step 5: Update v dan xyz dengan numerical integration
        self.v = self.v + v_dot*self.delta_t
        self.vx, self.vy, self.vz = self.v.flatten()
        # self.vx = self.v[0][0]
        # self.vy = self.v[1][0]
        # self.vz = self.v[2][0] #put values into vz variable for state feedback next loop
        self.pos = self.pos + self.v*self.delta_t
        self.x, self.y, self.z = self.pos.flatten()
        # self.x = self.pos[0][0]
        # self.y = self.pos[1][0]
        # self.z = self.pos[2][0]


        #Step 6: Update omega dan rpy
        self.w = self.w + w_dot*self.delta_t
        self.wx, self.wy, self.wz = self.w.flatten()
        # self.wx = self.w[0][0]
        # self.wy = self.w[1][0]
        # self.wz = self.w[2][0]
        ypr_dot = self.w_to_ypr_dot (self.w, self.roll, self.pitch, self.yaw)
        # print ("ypr dot", ypr_dot)
        self.rpy_dot = np.array([ypr_dot[2], ypr_dot[1], ypr_dot[0]])
        self.roll_dot, self.pitch_dot, self.yaw_dot = self.rpy_dot.flatten()
        # self.roll_dot = self.rpy_dot[0][0]
        # self.pitch_dot = self.rpy_dot[1][0]
        # self.yaw_dot = self.rpy_dot[2][0]
        # print ("rpy dot:", rpy_dot)
        self.rpy = self.rpy + self.rpy_dot*self.delta_t
        self.rpy = self.clip_angle(self.rpy)
        self.roll, self.pitch, self.yaw = self.rpy.flatten()
        # self.roll = self.rpy[0][0]
        # self.pitch = self.rpy[1][0]
        # self.yaw = self.rpy[2][0]

        #Step 7: Appending values
        self.v_values.append(self.v.flatten())
        self.pos_values.append(self.pos.flatten())
        self.w_values.append(self.w.flatten())
        self.rpy_values.append(self.rpy.flatten())
        self.rpy_dot_values.append(self.rpy_dot.flatten())
        self.u_values.append(self.u.flatten())
        self.tau_values.append(self.tau.flatten())

        #Step 8: Calculating error
        self.state_feedback_error = self.r - self.z  
        self.state_feedback_error_values.append(self.state_feedback_error)

        # pd_roll_error = self.desired_roll - self.roll
        self.pd_pitch_error = self.desired_pitch - self.pitch
        self.pd_controller_error_values.append(self.pd_pitch_error)

        # pd_roll_dot_error = self.desired_roll_dot - self.roll_dot
        self.pd_pitch_dot_error = self.desired_pitch_dot - self.pitch_dot
        self.pd_controller_dot_error_values.append(self.pd_pitch_dot_error)
                
        ##################
        self.state = [self.pitch, self.pitch_dot, self.wy]
        self.full_state = [self.x, self.y, self.z, self.vx, self.vy, self.vz, 
                           self.roll, self.pitch, self.yaw, self.roll_dot, self.pitch_dot, self.yaw_dot, self.wx, self.wy, self.wz]

        ########unmodified 
        done = bool(
            self.steps_left<0
        )

        if (abs(self.roll_dot) > 20 or
            abs (self.pitch_dot) > 20 or
            abs (self.yaw_dot) > 20):
            done = True
              
        self.pitch_reward = -np.square(self.desired_pitch-self.state[0])

        self.reward = self.pitch_reward #negative reward, orientasi pitch
        # print ("Reward:", self.reward)
        if not done:
            self.steps_left = self.steps_left-1
        
        self.cur_reward = self.reward
        self.cur_done = done
        return np.array([self.state]), self.reward, done, {} #state RL

    def reset(self):
        ### List initialization
        self.pos_values, self.v_values, self.w_values, self.rpy_values, self.rpy_dot_values = ([] for _ in range(5))
        self.u_values = []
        self.tau_values = []
        self.state_feedback_error_values = []
        self.pd_controller_error_values = []
        self.pd_controller_dot_error_values = []

        #Initial conditions
        _, self.initial_pitch = self._generate_initial_value()

        self.x, self.y, self.z = 0, 0, -1
        self.vx, self.vy, self.vz = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, self.initial_pitch, 0 #-0.5 sampai 0.5, step 0.1
        # print("reset initial roll: ", self.roll)
        self.roll_dot, self.pitch_dot, self.yaw_dot = 0, 0, 0
        self.wx, self.wy, self.wz = 0, 0, 0

        #Desired value
        self.r = -1.5
        # self.desired_roll = 0
        self.desired_pitch = 0
        # self.desired_roll_dot = 0
        self.desired_pitch_dot = 0

        #Storing to array
        self.pos = np.array([[self.x], [self.y], [self.z]])
        self.v = np.array([[self.vx], [self.vy], [self.vz]])
        self.w = np.array([[self.wx], [self.wy], [self.wz]])
        self.rpy = np.array([[self.roll], [self.pitch], [self.yaw]])
        self.rpy_dot = np.array([[self.roll_dot], [self.pitch_dot], [self.yaw_dot]])

        #Storing array elements to list -> dicomment berarti hrs print manual initial cond nya
        self.pos_values.append(self.pos.flatten())
        self.v_values.append(self.v.flatten())
        self.w_values.append(self.w.flatten())
        self.rpy_values.append(self.rpy.flatten())
        self.rpy_dot_values.append(self.rpy_dot.flatten())

        #Initializing control signals
        self.u = np.array([[0]])
        self.u_values.append(self.u.flatten())
        self.tau_x = self.tau_y = self.tau_z = 0
        self.tau = np.array([[self.tau_x], [self.tau_y], [self.tau_z]])
        self.tau_values.append(self.tau.flatten())
        
        #Initializing error
        self.state_feedback_error = self.r - self.z  
        self.state_feedback_error_values.append(self.state_feedback_error)

        # pd_roll_error = self.desired_roll - self.roll
        self.pd_pitch_error = self.desired_pitch - self.pitch
        self.pd_controller_error_values.append(self.pd_pitch_error)

        # pd_roll_dot_error = self.desired_roll_dot - self.roll_dot
        self.pd_pitch_dot_error = self.desired_pitch_dot - self.pitch_dot
        self.pd_controller_dot_error_values.append(self.pd_pitch_dot_error)

        self.full_state = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch,
                                    self.yaw, self.roll_dot, self.pitch_dot, self.yaw_dot, self.wx, self.wy, self.wz])
        
        self.state = np.array ([self.pitch, self.pitch_dot, self.wy])

        self.steps_left = self.MAX_EPISODE

        return np.array([self.state]) #return state RL aja: roll pitch, turunan roll pitch, omega x y

    def _generate_initial_value(self):
        possible_values = np.linspace(-0.5, 0.5, 11)
        possible_values = possible_values[possible_values!=0]
        initial_roll = random.choice(possible_values)
        initial_pitch = random.choice(possible_values)
        return initial_roll, initial_pitch


    def calculate_inertia_matrix(self):
        J = np.array([
            [self.Ixx, self.Ixy, self.Ixz],
            [self.Iyx, self.Iyy, self.Iyz],
            [self.Izx, self.Izy, self.Izz]
            ])
        return J
    
    
    def compute_state_feedback (self, A_mat, B_mat, C_mat, D_mat, desired_poles):
        E = ct.ctrb(A_mat, B_mat) #Controllability matrix
        #Verification:
        # print("Controllability matrix rank: ", np.linalg.matrix_rank(E))
        # print(E)
        if np.linalg.matrix_rank(E) != A_mat.shape[0]:
            raise ValueError("The system is not controllable")

        K = ct.place(A_mat, B_mat, desired_poles)
        # print("Gain: ", self.K)

        A_mat_new = A_mat - np.dot(B_mat, K)
        s = ct.poles(ct.ss(A_mat_new, B_mat, C_mat, D_mat))
        # print ("Poles for verification: ", s)

        # augmented matrix for solving Nx and Nu
        ss_matrix = np.block ([[A_mat, B_mat], [C_mat, D_mat]])
        # print (ss_matrix)
        # right hand side
        steady_state_vector = np.block([[np.zeros((A_mat.shape[0], 1))], 
                                        [np.eye(C_mat.shape[0])]])
        # print (steady_state_vector)

        N = np.linalg.solve(ss_matrix, steady_state_vector)
        Nx = N[:A_mat.shape[0], :]
        Nu = N[A_mat.shape[0]:, :]
        # print("N= ", N)
        # print("Nx = ", self.Nx)
        # print("Nu = ", self.Nu)
        return K, Nx, Nu
    
    #Roll pitch yaw to rotation matrix conversion
    def rpy_to_rot (self, roll, pitch, yaw):
        rot_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]])
        rot_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]])
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]])
        R = rot_z @ rot_y @ rot_x
        # print ("Rotation matrix:\n", R)
        return R
    
    def w_to_ypr_dot (self, w, roll, pitch, yaw):
        E = np.array([
            [-np.sin(pitch), 0, 1],
            [np.cos(pitch)*np.sin(roll), np.cos(roll), 0],
            [np.cos(pitch)*np.cos(roll), -np.sin(pitch), 0]])
        ypr_dot = E @ w
        return ypr_dot
    
    def clip_angle (self, angle):
        return np.mod(angle + np.pi, 2*np.pi) - np.pi

    
    #PD controller function
    def pd_controller(self, desired_angle, current_angle, desired_rate, current_rate, Kp, Kd):
        error = desired_angle - current_angle
        error_rate = desired_rate - current_rate
        torque = Kp*error + Kd*error_rate
        return torque
    
    def get_w_bar_sq (self, thrust_torque_vector):
        P = np.array([
            [-self.b, -self.b, -self.b, -self.b],
            [0, -(self.d)*self.b, 0, self.d*self.b],
            [self.d*self.b, 0, -(self.d)*self.b, 0],
            [self.k, -self.k, self.k, -self.k]
            ]) #Control allocation matrix
        P_inv = np.linalg.inv(P)
        w_bar_sq_vector = P_inv @ thrust_torque_vector
        return w_bar_sq_vector

############################# belum diubah
    
    def render(self):
        pass
