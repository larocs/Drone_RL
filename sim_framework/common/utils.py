import numpy as np
import random

def _set_discretized_uniform_position(class_):
    ang_pos = np.zeros(3)
    for i in range(3):
        ang_pos[i] =  class_.ang_ticks[np.random.choice(len(class_.ang_ticks), 1)]
    # self.agent.set_orientation(np.round(ang_pos,2).tolist())

    pos = np.zeros(3)
    for i in range(2):
        pos[i] = class_.x_y_ticks[np.random.choice(len(class_.x_y_ticks), 1)]

    pos[2] = class_.z_ticks[np.random.choice(len(class_.z_ticks), 1)] 
    # self.agent.set_position(np.round(pos,2).tolist())
    return pos, ang_pos

def _set_gaussian_position():
    # position
    pos = np.zeros(3)
    for i in range(3):
        z = random.gauss(0, 0.3)
        if i <= 1:
            pos[i] = z
        else:
            pos[i] = z + 1.7
    # angular pos
    ang_pos = np.zeros(3)
    for i in range(3):
        z = random.gauss(0, 0.6)
        ang_pos[i] = z
    return pos, ang_pos


def _set_uniform_position():

    # position
    pos = np.zeros(3)
    raio_init = 0.5
    for i in range(3):
        if i <= 1:
            z = random.uniform(-raio_init, raio_init)
            pos[i] = z
        else:
            z = random.uniform(1.7 - raio_init, 1.7 + raio_init)
            pos[i] = z

    # angular pos
    ang_pos = np.zeros(3)
    ang_max = 1.57  # 90 graus
    for i in range(3):
        z = random.uniform(-ang_max, ang_max)
        ang_pos[i] = z

    return pos, ang_pos

      

def _set_pose(self, position, orientation):
    self.agent.set_orientation(np.round(orientation,2).tolist())
    self.agent.set_position(np.round(position,2).tolist())
