import numpy as np


## Dicts

weight_dict_normal

weight_dict_normal = {}
weight_dict_normal['r_alive'] = 4.0        
weight_dict_normal['radius'] = -1.25
weight_dict_normal['pitch'] = 0
weight_dict_normal['yaw'] = 0
weight_dict_normal['roll'] = 0
weight_dict_normal['pitch_vel'] = 0
weight_dict_normal['yaw_vel'] = 0
weight_dict_normal['roll_vel'] = 0
weight_dict_normal['norm_a'] = 0
weight_dict_normal['std_a'] = 0
weight_dict_normal['death'] = 0
weight_dict_normal['lin_x_vel'] = 0
weight_dict_normal['lin_y_vel'] = 0
weight_dict_normal['lin_z_vel'] = 0
weight_dict_normal['integrative_error_x'] = 0
weight_dict_normal['integrative_error_y'] = 0
weight_dict_reward_24['integrative_error_z'] = 0




weight_dict_reward_24 = {}
weight_dict_reward_24['r_alive'] = 1.5        
weight_dict_reward_24['radius'] = -1.00
weight_dict_reward_24['pitch'] = 0
weight_dict_reward_24['yaw'] = 0
weight_dict_reward_24['roll'] = 0
weight_dict_reward_24['pitch_vel'] = -0.05
weight_dict_reward_24['yaw_vel'] = -0.1
weight_dict_reward_24['roll_vel'] = -0.05
weight_dict_reward_24['norm_a'] = 0
weight_dict_reward_24['std_a'] = 0
weight_dict_reward_24['death'] = 0
weight_dict_reward_24['lin_x_vel'] = 0
weight_dict_reward_24['lin_y_vel'] = 0
weight_dict_reward_24['lin_z_vel'] = 0
weight_dict_reward_24['integrative_error_x'] = 0
weight_dict_reward_24['integrative_error_y'] = 0
weight_dict_reward_24['integrative_error_z'] = 0



def reward(list_, raio,yaw,pitch, roll,\
                    yaw_vel, pitch_vel, roll_vel,lin_x_vel, lin_y_vel, lin_z_vel, norm_a, std_a,integrative_error, info=None):


    calc_dict = dict(zip(keys, list_))

    
    calculated_reward=0
    for element in calc_dict.values():
        calculated_reward += element

    return calculated_reward , calc_dict


keys = ["r_alive","radius","pitch","yaw","roll","pitch_vel","yaw_vel","roll_vel","lin_x_vel", \
    "lin_y_vel","lin_z_vel","norm_a","std_a","death","integrative_error_x","integrative_error_y","integrative_error_z"]

## The others 23 functions were gone because of refactoring


import_dict= {  'Normal' : { 'weight_list' : weight_dict_normal},
                'Reward_24' : { 'weight_dict' : weight_dict_reward_24}}


