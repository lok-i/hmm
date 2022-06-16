from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np

# environment config and setup

env_conf = {
            'set_on_rack': True,
            'render':True,
            'model_name': 'default_humanoid',
            'mocap':False,
            'observations':
            {
                'current_model_state': None
            },
            'actions':
            {   'joint_torques':
                    {
                        'dim': 15,
                        'torque_max': 5
                    }                
            },
            'rewards':
            {
                'zero_reward':None
            },
            'terminations':
            {
                'indefinite':None
            }                
            }

env = BipedEnv(**env_conf)

# Joint level PD controller setup
# torque = Kp ( q_des - q) + Kd (dq_des - dq)
pd_controller = PDController(
                             kps=np.full(env.n_act_joints,10.),
                             kds=np.full(env.n_act_joints,0.1),
                             )

q_act_des = np.zeros(env.n_act_joints)
dq_act_des = np.zeros(env.n_act_joints)

# loggers for traking plot
q_des_list = []
q_list = []
tau_list = []


# select the joint to test
joint_actuator_to_chk = "right_leg/hip_x"
actuator_id_being_chkd = env.model.actuator_name2id(joint_actuator_to_chk) 
base_dof = env.sim.data.qpos.shape[0] - env.n_act_joints


env.reset()

# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True

for _ in range(2000):

    qpos = env.sim.data.qpos[:].copy()
    qvel = env.sim.data.qvel[:].copy()
    q_act_des[actuator_id_being_chkd ] = np.radians(50*np.sin(0.01*_))
    torque = pd_controller.get_torque(
                                      q_des = q_act_des,
                                      dq_des= dq_act_des,
                                      q = qpos[base_dof :].copy(),
                                      dq= qvel[qvel.shape[0] - env.n_act_joints :].copy()
                                      )

    q_des_list.append(q_act_des[actuator_id_being_chkd])
    q_list.append(qpos[base_dof+actuator_id_being_chkd])
    tau_list.append(torque[actuator_id_being_chkd])

    obs,reward,done,info = env.step(action = torque)
    
time_steps = env.dt*np.arange(_+1)
plt.title("PD Tracking at joint: "+joint_actuator_to_chk)
plt.plot(time_steps,np.degrees(q_des_list),label='reference trajectory')
plt.plot(time_steps,np.degrees(q_list),label='ground truth')
plt.ylabel('Joint angle (in Degrees)')
plt.xlabel('Time')

plt.legend()
plt.grid()

plt2 = plt.twinx()
plt2.plot(time_steps,tau_list,color='green')
plt2.set_ylabel('Applied torque (in Nm)')

plt.show()
env.close()