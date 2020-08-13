from itertools import permutations, combinations, combinations_with_replacement
import itertools
import random
import math
import os

import numpy as np
import pandas as pd
import pickle

from pyrep import PyRep
from os.path import dirname, join, abspath
import pyrep.backend.sim  as sim
from pyrep.objects.shape import Shape

from sim_framework.envs.drone import Drone
# import sim_framework.reward_functions.rewards as rew_functions
import sim_framework.reward_functions.rewards as rew_functions2

from sim_framework.common import utils

from functools import partial 





class DroneEnv(object):

    def __init__(self, random, headless=True, use_vision_sensor = False, seed = 42, state = "Normal", SCENE_FILE=None,
        reward_function_name = 'Normal', buffer_size=None, neta=0.9 ,restart = False):

        if reward_function_name not in list(rew_functions2.import_dict.keys()):
            print("Wrong parameter passed on 'reward_function_name. Must be one of these: ", list(rew_functions.import_dict.keys()))

        if SCENE_FILE == None:
            if (use_vision_sensor):

                SCENE_FILE = join(dirname(abspath(__file__))) + '/../../scenes/ardrone_modeled_headless_vision_sensor.ttt' ## FIX
            else:

                SCENE_FILE = join(dirname(abspath(__file__))) + '/../../scenes/ardrone_modeled_headless.ttt' ## FIX
        
        assert state in ['Normal', 'New_Double', 'New_action']
        assert random in [False, 'Gaussian', 'Uniform','Discretized_Uniform'], \
                    "random should be one of these values [False, 'Gaussian', 'Uniform', 'Discretized_Uniform]"
        # assert random in [False, 'Gaussian', 'Uniform','Weighted','Discretized_Uniform'], \
                    # "random should be one of these values [False, 'Gaussian', 'Uniform', 'Weighted','Discretized_Uniform]"
      
        ## Setting Pyrep
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()
        self.agent = Drone(count=0, name="Quadricopter", num_joints=4, use_vision_sensor=use_vision_sensor)
        self.agent.set_control_loop_enabled(False) ## Disables the built-in PID control on V-REP motors
        #self.agent.set_motor_locked_at_zero_velocity(True)  ## When the force is set to zero, it locks the motor to prevent drifting
        self.agent.set_motor_locked_at_zero_velocity(False)  ## When the force is set to zero, it locks the motor to prevent drifting
        self.target = Shape('Quadricopter_target')


         ##Attributes

        self.action_space = self.agent.action_space

        if state == 'New_action':
            self.observation_space = 22
        elif state == 'New_Double':
            self.observation_space = 36
        else:
            self.observation_space = self.agent.observation_space
        self.restart=restart
        self.random=random
        self.dt = np.round(sim.simGetSimulationTimeStep(),4) ## timestep


        ## Integrative buffer
        self.buffer_size = buffer_size
        # if self.buffer_size:
        #     assert (isinstance(buffer_size,int)) and (buffer_size < 100)
        #     self.integrative_buffer_size = self.buffer_size
        #     self.integrative_buffer = np.empty((self.buffer_size, 3)) # 3 because its [x,y,z] or [roll,pitch,yaw]
        #     self.neta = neta



        self.state = state
        ## initial observation
        self._initial_state()
        self.first_obs = True ## hack so it doesn't calculate it at the first time
        self._make_observation()
        self.last_state=self.observation[:18]

        self.weighted = False

        # Creating lists
        if self.random == 'Discretized_Uniform':
            self._create_discretized_uniform_list()
        self.ptr=0
        # self._creating_initialization_list()


        ## Setting seed
        self.seed = seed
        self._set_seed(self.seed, self.seed)

        ## Reward Functions
        self.reward_function = partial(rew_functions2.reward, rew_functions2.import_dict[reward_function_name]['weight_list'])
        keys = ["r_alive","radius","pitch","yaw","roll","pitch_vel","yaw_vel","roll_vel","lin_x_vel","lin_y_vel","lin_z_vel","norm_a", \
                        "std_a","death","integrative_error_x","integrative_error_y","integrative_error_z"]
        self.weight_dict = dict(zip(keys, rew_functions2.import_dict[reward_function_name]['weight_list']))


    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def _make_observation(self):

        lst_o = []

        # Pose of the drone
        drone_pos = self.agent.get_position(relative_to=None)
        rel_orient = self.agent.get_orientation(relative_to=self.target)
        global_orient = self.agent.get_orientation(relative_to=None)
        lin_vel, ang_vel = self.agent.get_velocity()

        # Pose of the target
        target_pos = self.target.get_position(relative_to=None)
        goal_lin_vel, goal_ang_vel = self.agent.get_velocity(
            self.target.get_handle())

        # Relative pos:
        rel_pos = self.agent.get_position(relative_to=self.target)
        rel_ang_vel = ang_vel

        # Rotation matrix calculation (drone -> world)
        g11, g12, g13, g21, g22, g23, g31, g32, g33 = self.agent._rotatation_drone_world(
            global_orient)

        # State of the environment
        lst_o += list(rel_pos)
        lst_o += [g11, g12, g13, g21, g22, g23, g31, g32, g33]
        lst_o += rel_ang_vel
        lst_o += lin_vel

        # ## fifo
        # if self.buffer_size:
        #     self.integrative_buffer[:-1] = self.integrative_buffer[1:]; self.integrative_buffer[-1] = rel_ang_vel ## FIFO
        #     self.integrative_error = self._calc_accum_error(self.integrative_buffer, neta=self.neta)

        ## Actual State
        if self.state == 'New_action':
            if not self.first_obs:
                lst_o += list(self.current_action)
            else:
                lst_o += [0,0,0,0]


        self.observation = np.array(lst_o).astype('float32')
        if not self.first_obs:
            
            if self.state == 'New_Double':
                self.observation = np.append(self.observation[:18], self.last_state)	
            # Relative angular acceleration
            rel_ang_acc = ((self.observation[12:15] - self.last_state[12:15]) / self.dt)
            # Relative linear acceleration
            lin_acc = (self.observation[15:18] - self.last_state[15:18]) / self.dt
        else:
            if self.state == 'New_Double':
                self.observation = np.append(self.observation, self.observation)	

        self.first_obs=False

    def _set_seed(self,random_seed, numpy_seed):
        random.seed(a = random_seed)
        np.random.seed(seed=numpy_seed)
        
    def _make_action(self, a):

        
        self.agent.set_thrust_and_torque(a, force_zero=False)
        self.current_action=a

    def step(self, action):

        if isinstance(action, dict):
            ac = np.zeros(4)
            for i in range(4):
                ac[i] = action['action{}'.format(i)]
            action = ac

        # Clipping the action taken
        action = np.clip(action, 0, self.agent.joints_max_velocity)

        # Actuate
        self._make_action(action)

        # Step
        self.pr.step()  # Step the physics simulation


        self.last_state = self.observation[:18]
        # Observe
        self._make_observation()

        # Reward
        drone_pos, drone_orient, yaw,pitch, roll,yaw_vel,pitch_vel, roll_vel,lin_x_vel, lin_y_vel, lin_z_vel, \
                norm_a, std_a = self._get_reward_data()
        reward, reward_dict = self.reward_function(self, self.radius ,yaw,pitch, roll, 
                                    yaw_vel,pitch_vel, roll_vel,lin_x_vel, lin_y_vel, lin_z_vel,
                                    norm_a, std_a ,self.integrative_error)
        info = reward_dict

        # Check if state is terminal
        if self.weighted:
            stand_threshold = 11
        elif self.random == 'Discretized_Uniform':
            stand_threshold = 6.5
        else:
            stand_threshold = 3.2

        done = (self.radius > stand_threshold)
        if done:
            reward += self.weight_dict['death']
        
        return self.observation, reward, done, info

    def _get_reward_data(self):

        drone_pos = self.agent.get_position(relative_to=self.target)
        drone_orient = self.agent.get_orientation(relative_to=self.target)
        self.drone_orientation=drone_orient

        roll = drone_orient[0]
        pitch = drone_orient[1]
        yaw = drone_orient[2]

        roll_vel, pitch_vel, yaw_vel = self.observation[12:15]
        lin_x_vel, lin_y_vel, lin_z_vel = self.observation[15:18]
    
        self.radius = math.sqrt(drone_pos[0]**2 + drone_pos[1]**2 + drone_pos[2]**2)

        lin_x_vel, lin_y_vel, lin_z_vel = self.observation[15:18]
        norm_a = np.linalg.norm(self.current_action,ord=2)
        std_a = self.current_action.std()

        if not self.buffer_size:
            self.integrative_error = None
        return drone_pos, drone_orient, yaw,pitch, roll, yaw_vel,pitch_vel, roll_vel,lin_x_vel, lin_y_vel, lin_z_vel,norm_a, std_a

    def reset(self):

        ## Zeroing the integrative buffer:
        if self.buffer_size:
            self.integrative_buffer[:] = np.nan

        self.current_action = np.array([0,0,0,0])

        
        if self.restart == True:
                self.pr.stop()
                self._reset_position(how=self.random)
                self.pr.start()
        else:
                self._reset_position(how=self.random)


        self.first_obs=True
        self._make_observation()
        self.last_state = self.observation[:18]

        return self.observation

    def _reset_position(self, how = False):


        # self.pr.step()
        # self.agent.set_orientation([-0,0,-0])

        if how == "Gaussian":
            # self._set_gaussian_position()
            position, orientation = utils._set_gaussian_position()
            self._set_pose(position, orientation)
            
        elif how == 'Uniform':
            position, orientation = utils._set_uniform_position()
            self._set_pose(position, orientation)
            # self._set_uniform_position()
        elif how == 'Discretized_Uniform':
            position, orientation = utils._set_discretized_uniform_position(self)
            self._set_pose(position, orientation)
        elif how == 'Weighted':
            raise NotImplementedError
            # if self.weighted == False:
            #     self.weighted = True
            # else:
            #     pass
            # chosen_position, chosen_orientation = self._sampling_weighted_multinomial()
            # self.agent.set_position(chosen_position)
            # self.agent.set_orientation(chosen_orientation.tolist())
                
        else:
            self._set_initial_position()

        
        self.agent.set_thrust_and_torque(np.asarray([0.] * 4), force_zero=True)
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.agent.set_joint_target_velocities(self.initial_joint_velocities)
        self.agent.set_joint_target_positions(self.initial_joint_target_positions)
    
    def _set_pose(self, position, orientation):
        self.agent.set_orientation(np.round(orientation,2).tolist())
        self.agent.set_position(np.round(position,2).tolist())

    def _set_initial_position(self):
        self.agent.set_position(self.initial_position)
        self.target.set_position(self.target_initial_position)
        self.agent.set_orientation(self.initial_orientation)
        self.target.set_orientation(self.target_initial_orientation)
        # self.agent.set_joint_positions(self.initial_joint_positions)
 
    def _create_discretized_uniform_list(self):
        
        num_discretization = 7
        bound_of_distribuition = 1.5
        size=2
        self.x_y_ticks = np.round(np.linspace(-bound_of_distribuition,bound_of_distribuition,num=num_discretization),2)

        num_discretization = 11
        bound_of_distribuition = 0.5
        size=1
        z_ticks = np.round(np.linspace(-bound_of_distribuition,bound_of_distribuition,num=num_discretization),2)
        self.z_ticks = (z_ticks+1.7)

        num_discretization = 11
        bound_of_distribuition = 1.57/2
        size=3
        self.ang_ticks = np.round(np.linspace(-bound_of_distribuition,bound_of_distribuition,num=num_discretization),2)


    def _initial_state(self):

        self.initial_joint_positions = self.agent.get_joint_positions()
        self.initial_position = self.agent.get_position()
        self.initial_orientation = self.agent.get_orientation()
        self.drone_orientation=self.initial_orientation
        self.target_initial_position = self.target.get_position()
        self.target_initial_orientation = self.target.get_orientation()
        self.initial_joint_velocities = self.agent.get_joint_velocities()
        self.initial_joint_target_velocities = self.agent.get_joint_target_velocities()
        self.initial_joint_target_positions = self.agent.get_joint_target_positions()
        self.current_action = np.array([0,0,0,0])

    # def _sampling_weighted_multinomial(self,):
        
    #     ## orientation
    #     experiment = np.random.multinomial(n=1,pvals= self._prob_array ,size =1) 
    #     chosen_index = np.argmax(experiment)
    #     chosen_orientation = self._orientation_permutations[chosen_index-1]
        
    #     ## position
    #     experiment = np.random.choice(np.arange(len(self.position_permutations)), size=None, replace=True, p=None)
    #     # experiment = np.random.multinomial(n=1,pvals= nplen(self._prob_array) ,size =1) 
    #     chosen_position = (self.position_permutations[experiment])
    #     return chosen_position, chosen_orientation

