from re import A
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np

env = BipedEnv(model_name='humanoid_no_hands')
done = False

pd_controller = PDController(
                             kps=np.full(env.n_act_joints,10.),
                             kds=np.full(env.n_act_joints,0.1),
                             )

env.reset()
if env.is_render:
    # env.viewer._hide_overlay = True
    env.viewer._paused = True

q_act_des = np.zeros(env.n_act_joints)
dq_act_des = np.zeros(env.n_act_joints)


q_des_list = []
q_list = []
tau_list = []


actuator_id_bing_chked = env.model.actuator_name2id("left_ankle_y") 
for _ in range(1000):

    qpos,qvel = env.get_state()
    q_act_des[actuator_id_bing_chked ] = np.radians(50*np.sin(0.01*_))
    
    torque = pd_controller.get_torque(
                                      q_des = q_act_des,
                                      dq_des= dq_act_des,
                                      q = qpos[qpos.shape[0] - env.n_act_joints :].copy(),
                                      dq= qvel[qvel.shape[0] - env.n_act_joints :].copy()
                                      )

    q_des_list.append(q_act_des[actuator_id_bing_chked])
    q_list.append(qpos[7+actuator_id_bing_chked])
    tau_list.append(torque[actuator_id_bing_chked])

    obs,reward,done,info = env.step(action = torque)
    env.render()
    


time_steps = env.dt*np.arange(_+1)
plt.plot(time_steps,np.degrees(q_des_list),label='desired')
plt.plot(time_steps,np.degrees(q_list),label='gt')
plt.legend()
plt.grid()
plt2 = plt.twinx()
plt2.plot(time_steps,tau_list,color='green')

plt.show()


env.close()