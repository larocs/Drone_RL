from pyrep.robots.robot_component import RobotComponent
from typing import List, Tuple, Union
from pyrep.objects.shape import Shape
from pyrep.objects.force_sensor import ForceSensor
import numpy as np
from gym import spaces
import pyrep.backend.sim  as sim
import math
from pyrep.objects.vision_sensor import VisionSensor

class Drone(RobotComponent):
    """Base class representing a drone.
    """

    def __init__(self, count: int, name: str, num_joints: int,
                 use_vision_sensor = False, base_name: str = None,
                 max_velocity=100):
        
        self.use_vision_sensor = use_vision_sensor

        self.oh_joint = ["Quadricopter_propeller_joint1", "Quadricopter_propeller_joint2",
                         "Quadricopter_propeller_joint3", "Quadricopter_propeller_joint4"]

        
        propeller_names = ["Quadricopter_propeller_respondable1",
                           "Quadricopter_propeller_respondable2",
                           "Quadricopter_propeller_respondable3",
                           "Quadricopter_propeller_respondable4"]

        self.propellers = [Shape(name) for name in propeller_names]

        force_sensor_names = ["Quadricopter_propeller1",
                              "Quadricopter_propeller2",
                              "Quadricopter_propeller3",
                              "Quadricopter_propeller4"]

        self.force_sensors = [ForceSensor(name) for name in force_sensor_names]

        super().__init__(count, name, self.oh_joint, base_name)

        self.joint_handles = [joint._handle for joint in self.joints]
        self.force_sensor_handles = [sensor.get_handle() for sensor in self.force_sensors]


        if self.use_vision_sensor:
            # vision_sensor_names = ["Vision_sensor_floor", 'Vision_sensor_frontal']
            vision_sensor_names = [ 'Vision_sensor_frontal']

    
            self.vision_sensors =  [VisionSensor(name) for name in vision_sensor_names]

        # One action per joint
        num_act = len(self.oh_joint)

        # Multiple dimensions per shape
        #num_obs = ((len(self.oh_shape)*3*3)+1);
        num_obs = 18

        self.joints_max_velocity = max_velocity
        act = np.array([self.joints_max_velocity] * num_act)
        obs = np.array([np.inf] * num_obs)

        self.action_space = spaces.Box(np.array([0]*num_act), act)
        self.observation_space = spaces.Box(-obs, obs)

    def _my_simTransformVector(self,matrix,vector):
        vector = np.asarray(vector)
        matrix = np.asarray(matrix).reshape(3,4)
        matrix = np.delete(matrix, -1, axis=1)
        return np.multiply(vector,matrix).sum(axis=1)
    
    def get_velocity(self, handle=None) -> List[float]:
        """
        Retrieves the linear and/or angular velocity of an object, in absolute coordinates.
         The velocity is a measured velocity (i.e. from one simulation step to the next),
          and is available for all objects in the scene. 

        :return: A list containing the x, y, z position of the object.
        """
        return sim.simGetObjectVelocity(handle or self._handle)

    def _simAddForceAndTorque(self, handle, force, torque):
        sim.lib.simAddForceAndTorque(handle, force, torque)

    def read_force_sensor(self, handle) -> Tuple[List[float], List[float]]:
        """Reads the force and torque applied to a force sensor.
        :return: A tuple containing the applied forces along the
            sensor's x, y and z-axes, and the torques along the
            sensor's x, y and z-axes.
        """
        _, forces, torques = sim.simReadForceSensor(handle)
        return forces, torques

    def _rotatation_drone_goal(self, rel_orient):
        # Rotation matrix calculation (drone -> goal)
        r11 = math.cos(rel_orient[2])*math.cos(rel_orient[1])
        r12 = math.cos(rel_orient[2])*math.sin(rel_orient[1])*math.sin(
            rel_orient[0]) - math.sin(rel_orient[2])*math.cos(rel_orient[0])
        r13 = math.cos(rel_orient[2])*math.sin(rel_orient[1])*math.cos(
            rel_orient[0]) + math.sin(rel_orient[2])*math.sin(rel_orient[0])
        r21 = math.sin(rel_orient[2])*math.cos(rel_orient[1])
        r22 = math.sin(rel_orient[2])*math.sin(rel_orient[1])*math.sin(
            rel_orient[0]) + math.cos(rel_orient[2])*math.cos(rel_orient[0])
        r23 = math.sin(rel_orient[2])*math.sin(rel_orient[1])*math.cos(
            rel_orient[0]) - math.cos(rel_orient[2])*math.sin(rel_orient[0])
        r31 = -math.sin(rel_orient[1])
        r32 = math.cos(rel_orient[1])*math.sin(rel_orient[0])
        r33 = math.cos(rel_orient[1])*math.cos(rel_orient[0])
        return r11, r12, r13, r21, r22, r23, r31, r32, r33

    def _rotatation_drone_world(self, global_orient):
        # Rotation matrix calculation (drone -> world)

        g11 = math.cos(global_orient[2])*math.cos(global_orient[1])
        g12 = math.cos(global_orient[2])*math.sin(global_orient[1])*math.sin(
            global_orient[0]) - math.sin(global_orient[2])*math.cos(global_orient[0])
        g13 = math.cos(global_orient[2])*math.sin(global_orient[1])*math.cos(
            global_orient[0]) + math.sin(global_orient[2])*math.sin(global_orient[0])
        g21 = math.sin(global_orient[2])*math.cos(global_orient[1])
        g22 = math.sin(global_orient[2])*math.sin(global_orient[1])*math.sin(
            global_orient[0]) + math.cos(global_orient[2])*math.cos(global_orient[0])
        g23 = math.sin(global_orient[2])*math.sin(global_orient[1])*math.cos(
            global_orient[0]) - math.cos(global_orient[2])*math.sin(global_orient[0])
        g31 = -math.sin(global_orient[1])
        g32 = math.cos(global_orient[1])*math.sin(global_orient[0])
        g33 = math.cos(global_orient[1])*math.cos(global_orient[0])
        return g11, g12, g13, g21, g22, g23, g31, g32, g33

    def set_thrust_and_torque(self, forces, torques=None, force_zero=None):
        '''Sets force (thrust) and torque to a respondable shape.\
        Must be configured specifically according to each robot's dynamics.

        Args:
            forces: Force vector applied at each propeller.
            torques: Torque vector applied at each propeller.
        '''
        # torques = 0.00002*forces

        count = 1  # set torque's clockwise (even) and counter-clockwise (odd)
        
        t = sim.simGetSimulationTime()

        # for propeller, joint, pwm, torque in zip(self.propellers,self.joints, forces, torques):
        for k, (propeller, joint, pwm) in enumerate(zip(self.propellers,self.joints, forces)):

            # for propeller, force, torque in zip(self.force_sensors, forces, torques):

            force = 1.5618e-4*pwm*pwm + 1.0395e-2*pwm + 0.13894
            if force_zero:
                force = 0

            rot_matrix = (sim.simGetObjectMatrix(self.force_sensor_handles[k],-1))

            rot_matrix[3] = 0
            rot_matrix[7] = 0
            rot_matrix[11] = 0
            
            z_force = np.array([0.0, 0.0, force])

            applied_force = list(self._my_simTransformVector(rot_matrix, z_force))

            if count % 2:
                z_torque = np.array([0.0, 0.0, -0.001*pwm])

            else:
                z_torque =np.array([0.0, 0.0, 0.001*pwm]) ## DEIXANDO IGUAL AO V-REP


            applied_torque = list(self._my_simTransformVector(rot_matrix, (z_torque)))


            count += 1
            self._simAddForceAndTorque(
                propeller._handle, applied_force, applied_torque)

            joint.set_joint_position(position = t*10, allow_force_mode=False) ## allow_force_mode = True has a built-in step inside
