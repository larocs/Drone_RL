import numpy as np




def reward( list_,self, raio,yaw,pitch, roll,\
                    yaw_vel, pitch_vel, roll_vel,lin_x_vel, lin_y_vel, lin_z_vel,
                     norm_a, std_a,integrative_error, info=None):


    weight_dict = dict(zip(keys, list_))

    
    calc_dict = {}
    calc_dict['r_alive'] = weight_dict['r_alive']
    calc_dict['radius'] = (weight_dict['radius']*raio)
    calc_dict['pitch'] = (weight_dict['pitch']*abs(pitch))
    calc_dict['yaw'] = (weight_dict['yaw']*abs(yaw))
    calc_dict['roll'] = (weight_dict['roll']*abs(roll))
    calc_dict['pitch_vel'] = (weight_dict['pitch_vel']*abs(pitch_vel))
    calc_dict['yaw_vel'] = (weight_dict['yaw_vel']*abs(yaw_vel))
    calc_dict['roll_vel'] = (weight_dict['roll_vel']*abs(roll_vel))
    calc_dict['lin_x_vel'] = (weight_dict['lin_x_vel']*abs(lin_x_vel))
    calc_dict['lin_y_vel'] = (weight_dict['lin_y_vel']*abs(lin_y_vel))
    calc_dict['lin_z_vel'] = (weight_dict['lin_z_vel']*abs(lin_z_vel))
    calc_dict['norm_a'] = (weight_dict['norm_a']*abs(norm_a))
    calc_dict['std_a'] = (weight_dict['std_a']*abs(std_a))
    if integrative_error != None:
        calc_dict['integrative_error_x'] = (weight_dict['integrative_error_x']*abs(integrative_error[0]))
        calc_dict['integrative_error_y'] = (weight_dict['integrative_error_y']*abs(integrative_error[0]))
        calc_dict['integrative_error_z'] = (weight_dict['integrative_error_z']*abs(integrative_error[0]))
        
    
    calculated_reward=0
    for element in calc_dict.values():
        calculated_reward += element

    return calculated_reward , calc_dict


keys = ["r_alive","radius","pitch","yaw","roll","pitch_vel","yaw_vel","roll_vel","lin_x_vel", \
    "lin_y_vel","lin_z_vel","norm_a","std_a","death","integrative_error_x","integrative_error_y","integrative_error_z"]

## The others 23 functions were gone because of refactoring

import_dict= {  'Normal' : { 'weight_list' : [4.0, -1.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0,]},
                'Reward_24' : { 'weight_list' : [1.5, -1.00, 0, 0, 0, -0.05, -0.1, -0.05, 0, 0, 0, 0, 0, 0,0, 0, 0]}}


# weight_dict_normal = {}
# weight_dict_normal['r_alive'] = 4.0        
# weight_dict_normal['radius'] = -1.25
# weight_dict_normal['pitch'] = 0
# weight_dict_normal['yaw'] = 0
# weight_dict_normal['roll'] = 0
# weight_dict_normal['pitch_vel'] = 0
# weight_dict_normal['yaw_vel'] = 0
# weight_dict_normal['roll_vel'] = 0
# weight_dict_normal['norm_a'] = 0
# weight_dict_normal['std_a'] = 0
# weight_dict_normal['death'] = 0
# weight_dict_normal['lin_x_vel'] = 0
# weight_dict_normal['lin_y_vel'] = 0
# weight_dict_normal['lin_z_vel'] = 0
# weight_dict_normal['integrative_error_x'] = 0
# weight_dict_normal['integrative_error_y'] = 0
# weight_dict_reward_24['integrative_error_z'] = 0