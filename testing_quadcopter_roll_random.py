import gym
import math
from quadcopter_model_roll_random import Quadcopter_Roll_Random
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.ticker import MultipleLocator, FuncFormatter

random.seed(10)
env = Quadcopter_Roll_Random ()
observation = env.reset()
action = [[0, 0]]
reward = 0

# print ("Initial full state: ", env.full_state)
# print("Initial roll: ", env.full_state[6])

model = PPO.load("ppo_quadcopter_roll_random", env=env)

print ("Initial observation: ", observation)
action_array = []
time_array = []
observation_array =[]
reward_array = []
times = 0

action_array.append(action[0])
time_array.append(times)
observation_array.append(observation[0])
reward_array.append(reward)

for i in range(0,1000):
    action, _state = model.predict(observation[0], deterministic=True)
    # action = [5, 0.1]
    # action = [10, 0.5]
    # action = [15, 1]
    observation, reward, terminated, _ = env.step(action)
    # print ("Full state: ", env.full_state)
    # print("Roll: ", env.full_state[6])
    # print ("Observation: ", observation)
    env.render()
    action_array.append(action)
    time_array.append(times)
    observation_array.append(observation[0])
    reward_array.append(reward)

    times = times + env.delta_t
    if terminated:
        print('done', i)
        observation = env.reset()



env.close()

# print (env.pos_values) #check initial values included in array
print ("K = ", env.K)
print ("Nx = ", env.Nx)
print ("Nu =", env.Nu)

# print(env.tau_values)

# print ("Reward: ", reward_array)
# print("Obs: ", observation_array)
action_values = np.array(action_array)
print("Action: ", action_values)


reward_values = np.array(reward_array)
observation_values = np.array(observation_array)
print("Observation: ", observation_values)

env.v_values = np.array(env.v_values)
env.pos_values = np.array(env.pos_values)
env.w_values = np.array(env.w_values)
env.rpy_values = np.array(env.rpy_values)
env.rpy_dot_values = np.array(env.rpy_dot_values)
env.u_values = np.array(env.u_values)
env.tau_values = np.array(env.tau_values)

env.state_feedback_error_values = np.array(env.state_feedback_error_values)
env.pd_controller_error_values = np.array(env.pd_controller_error_values)
env.pd_controller_dot_error_values = np.array(env.pd_controller_dot_error_values)

# Functions to check control performance

# Function to verify if the system has reached steady state
def is_steady_state(response, tolerance, window_size): 
    steady_state_value = response[-1] # last value is steady state value
    # Calculate the absolute differences between the recent values and the steady state value
    # Recent values correspond to window size
    recent_differences = np.abs(response[-window_size:] - steady_state_value)
    # Calculate the tolerance band around the steady state value
    tolerance_band = tolerance*np.abs(steady_state_value)
    # Check if all recent differences are within the tolerance band
    within_tolerance = recent_differences <= tolerance_band
    return np.all(within_tolerance)

# Function to calculate steady state error of response
def calculate_steady_state_error(response, desired_value): 
    steady_state_value = response[-1]
    steady_state_error = np.abs(steady_state_value - desired_value)
    return steady_state_error

def calculate_settling_time(time, response, tolerance):
    # Steady state value
    steady_state_value = response[-1]
    # Calculate the absolute differences between the response and the steady state value
    differences = np.abs(response - steady_state_value) 
    # Tolerance band
    tolerance_band = tolerance * np.abs(steady_state_value)
    # Check if differences are in range of tolerance band
    within_tolerance = differences <= tolerance_band
    # Create array that contains times where the responses are within the tolerance band
    settling_time_array = np.where(within_tolerance)[0]
    
    # If no indices are found, return None
    if len(settling_time_array) == 0:
        return None
    
    # The settling time is the time at the first index of settling time aray
    settling_time = time[settling_time_array[0]]
    return settling_time

def calculate_rise_time(time, response, lower_threshold, upper_threshold, reference_value):
    if(reference_value < 0): #Ascending
        lower_value = lower_threshold*reference_value
        upper_value = upper_threshold*reference_value
        start_index = np.where(response >= lower_value)[0][0]
        end_index = np.where(response >= upper_value)[0][0]

    elif(reference_value > 0): #Descending
        lower_value = lower_threshold*reference_value
        upper_value = upper_threshold*reference_value
        start_index = np.where(response <= upper_value)[0][0]
        end_index = np.where(response <= lower_value)[0][0]
    
    rise_time = abs(time[end_index] - time[start_index])

    return rise_time

def calculate_squared_sum (array):
    squared_array = array ** 2
    sum_squared_array = np.sum(squared_array)
    return sum_squared_array

# Function to assist plotting
def format_fn(tick_val, tick_pos): #For y axis label
    if tick_val == 0:
        return "0"
    elif tick_val == 1:
        return r"$\pi$"
    elif tick_val == -1:
        return r"$-\pi$"
    else:
        rounded_val = round(tick_val, 2)
        return r"${0}\pi$".format(rounded_val)
    

#Converting to something times pi radian
rpy_values_pi = np.array(env.rpy_values) / np.pi
rpy_dot_values_pi = np.array(env.rpy_dot_values) / np.pi
w_values_pi = np.array(env.w_values) / np.pi
pd_controller_error_pi = np.array(env.pd_controller_error_values) / np.pi
pd_controller_dot_error_pi = np.array(env.pd_controller_dot_error_values) / np.pi

#Check if roll has reached steady state
if is_steady_state(rpy_values_pi[:,0], 0.02, 300):
    roll_steady_state_error = calculate_steady_state_error(rpy_values_pi[:,0], env.desired_roll/np.pi)
    print("The system's roll value has reached steady state condition.")
    print("Roll steady state error: ", roll_steady_state_error)
else:
    print("The system's roll value has not reached steady state condition.")

#Calculate roll rise time
roll_rise_time = calculate_rise_time(time_array, rpy_values_pi[:,0], 0.1, 0.5, rpy_values_pi[:, 0][0])
print("Roll rise time: ", roll_rise_time, " seconds")
#Calculate roll settling time
roll_settling_time = calculate_settling_time(time_array, rpy_values_pi[:,0], 0.02)
print("Roll settling time: ", roll_settling_time, " seconds")
#Calculate sum of roll torque squared
tau_x_squared_sum = calculate_squared_sum(env.tau_values[:,0])
print ("Sum of squared roll torque: ", tau_x_squared_sum)

# Plotting
plt.figure()
plt.plot(time_array, action_values[:,0], color='r', label ='Kp roll')
plt.plot(time_array, action_values[:,1], color='g', label='Kd roll')
plt.title('Actions over Time')
plt.xlabel('Time (s)')
plt.ylabel('Actions')
plt.legend()

plt.figure()
plt.plot(time_array, reward_values)
plt.title ('Reward over Time')
plt.xlabel('Time(s)')
plt.ylabel('Reward')

#Not something times pi radian
# plt.figure()
# plt.plot(time_array, observation_values[:,0])
# plt.title ('Roll over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Roll (rad)')

# plt.figure()
# plt.plot(time_array, observation_values[:,1])
# plt.title('Roll Rate over Time')
# plt.xlabel('Time(s)')
# plt.ylabel('Roll Rate (rad/s)')

# plt.figure()
# plt.plot(time_array, observation_values[:,2])
# plt.title('x Angular Velocity')
# plt.xlabel('Time(s)')
# plt.ylabel('x Angular Velocity (rad/s)')
# plt.legend()

plt.figure()
plt.plot(time_array, env.u_values)
plt.title('State Feedback Control Signal over Time')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')

# plt.figure()
# plt.plot(time_array, env.tau_values[:, 0], color='r', label='Roll Torque')
# plt.plot(time_array, env.tau_values[:, 1], color='g', label='Pitch Torque')
# plt.plot(time_array, env.tau_values[:, 2], color='b', label='Yaw Torque')
# plt.title('PD Controller Control Signal over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Torque (Nm)')
# plt.legend()

plt.figure()
plt.plot(time_array, env.tau_values[:, 0])
plt.title('PD Controller Roll Control Signal over Time')
plt.xlabel('Time (s)')
plt.ylabel('Roll Torque (Nm)')


# # Plotting velocity and position
# plt.figure(figsize=(12, 6))

# # Plotting velocity
# plt.subplot(1, 2, 1)
# plt.plot(time_array, env.v_values[:, 0], color='r', label='vx')
# plt.plot(time_array, env.v_values[:, 1], color='g', label='vy')
# plt.plot(time_array, env.v_values[:, 2], color='b', label='vz')
# plt.title('Velocity over time')
# plt.xlabel('Time(s)')
# plt.ylabel('Velocity (m/s)')
# plt.legend()

# # Plotting position
# plt.subplot(1, 2, 2)
# plt.plot(time_array, env.pos_values[:, 0], color='r', label='x')
# plt.plot(time_array, env.pos_values[:, 1], color='g', label='y')
# plt.plot(time_array, env.pos_values[:, 2], color='b', label='z')
# plt.title('Position over time')
# plt.xlabel('Time(s)')
# plt.ylabel('Position (m)')
# plt.legend()

plt.figure()
plt.plot(time_array, env.pos_values[:, 2], color='r', label='z')
plt.title('z Position over Time')
plt.xlabel('Time(s)')
plt.ylabel('z Position (m)')

plt.figure()
plt.plot(time_array, env.v_values[:, 2], color='g', label='vz')
plt.title('Velocity of z over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity of z (m/s)')


# # Plotting angular velocity
# plt.figure()
# plt.plot(time_array, w_values_pi[:, 0], color='r', label='wx')
# plt.plot(time_array, w_values_pi[:, 1], color='g', label='wy')
# plt.plot(time_array, w_values_pi[:, 2], color='b', label='wz')
# plt.title('Angular Velocity over time')
# plt.xlabel('Time (s)')
# plt.ylabel('Angular Velocity (rad/s)')
# plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.5))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(format_fn))
# plt.legend()

# # Plotting rpy orientation
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.plot(time_array, rpy_values_pi[:, 0], color='r', label='roll')
# plt.plot(time_array, rpy_values_pi[:, 1], color='g', label='pitch')
# plt.plot(time_array, rpy_values_pi[:, 2], color='b', label='yaw')
# plt.title('RPY Orientation over time')
# plt.xlabel('Time (s)')
# plt.ylabel('RPY Orientation (rad)')
# plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.05))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(format_fn))
# plt.legend()

# # Plotting rpy dot
# plt.subplot (1, 2, 2)
# plt.plot(time_array, rpy_dot_values_pi[:, 0], color='r', label='roll dot')
# plt.plot(time_array, rpy_dot_values_pi[:, 1], color='g', label='pitch dot')
# plt.plot(time_array, rpy_dot_values_pi[:, 2], color='b', label='yaw dot')
# plt.title('RPY Dot over time')
# plt.xlabel('Time (s)')
# plt.ylabel('RPY Rate (rad/s)')
# plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.5))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(format_fn))
# plt.legend()

plt.figure()
plt.plot(time_array, rpy_values_pi[:, 0], color='r')
plt.title('Roll over Time')
plt.xlabel('Time(s)')
plt.ylabel('Roll (rad)')
plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.05))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_fn))

plt.figure()
plt.plot(time_array, rpy_dot_values_pi[:, 0], color='g')
plt.title('Roll Rate over Time')
plt.xlabel('Time(s)')
plt.ylabel('Roll Rate (rad/s)')
plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.5))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_fn))


# Plotting state feedback errors
plt.figure()
plt.plot(time_array, env.state_feedback_error_values, label='z Position Error')
plt.title('State Feedback Errors')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.legend()

# Plotting PD controller errors
plt.figure()
plt.plot(time_array, pd_controller_error_pi, color='r', label='Roll Error')
plt.title('PD Controller RPY Errors')
plt.xlabel('Time (s)')
plt.ylabel('Error (rad)')
plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.05))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_fn))
plt.legend()

plt.figure()
plt.plot(time_array, pd_controller_dot_error_pi, color='g', label='Roll Dot Error')
plt.title('PD Controller RPY Dot Errors')
plt.xlabel('Time (s)')
plt.ylabel('Error (rad/s)')
plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.5))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_fn))
plt.legend()

plt.tight_layout()
plt.show()