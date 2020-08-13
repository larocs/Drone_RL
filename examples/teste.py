
from sim_framework.envs.drone_env import DroneEnv
import numpy as np



# env = DroneEnv(random=False)
# env = DroneEnv(random='Discretized_Uniform', state='New_Double')
# env = DroneEnv(random='Uniform', state='New_Double')
env = DroneEnv(random='Gaussian', state='New_Double')




state = env.reset()
print(state.shape)
for i in range(20):
    a = np.asarray([0,0,0,0])
    next_state, reward, done, env_info = env.step(a)
    print(next_state.shape)
print('ouk')
env.shutdown()