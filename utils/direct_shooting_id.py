
  
#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#

from gym_hmm_ec.envs.bipedal_env import BipedEnv
from mujoco_py import functions


import casadi as cas
from casadi import *




class mujoco_model(cas.Callback):
    def __init__(self,
                    env,
                    N,
                    q0,
                    dq0,
                    u0,
                    ):
        Callback.__init__(self)
        self.env = env
        self.N = N
        self.q0 = q0
        self.dq0 = dq0
        self.u0 = u0
        

        self.construct("Mujoco_model", {"enable_fd": True})
        # print("\nInitialised\n")

    # def get_n_in(self):
    #     return self.u0.shape[0] 


    def get_sparsity_in(self, i):
        if i==0:
            return cas.Sparsity.dense(self.u0.shape[0]*N,1)

    def eval(self, args):
        # A list of DMs comes in
        # arg = np.array(args[0])
        
        # state to qpos and qvel
        arg = np.array(args[0]).T[0]
        # arg = args[0]
        # print(arg)
        # exit()

        self.env.sim.data.qpos[:] = self.q0 
        self.env.sim.data.qvel[:] = self.dq0 

        cost = 0
        for i in range(N-1):
            # for j in range(self.u0.shape[0]):
            #     print('LHS',self.env.sim.data.ctrl[j])
            #     print('RHS',arg[i*self.u0.shape[0] + j])
            #     self.env.sim.data.ctrl[j] = arg[i*self.u0.shape[0] + j  ]
            
            self.env.sim.data.ctrl[:] = arg[i*self.u0.shape[0] : (i+1)*self.u0.shape[0]  ]
            
            self.env.sim.step()
            for aj in arg[i*self.u0.shape[0] : (i+1)*self.u0.shape[0] ]:
                cost += (aj + 1)
    
        # print(cost)
        # TODO: 
        # add grfs
        
        # self.env.sim.step()
        # functions.mj_forward(env.model, env.sim.data)

        #  Some purely numeric code that is not supported by CasADi
        # Could also use if/while etc
        # res = np.concatenate([,self.env.sim.data.qacc]).ravel()#gamma(arg)

        # A list of DMs should go out
        results = [cost]
        return results


# environment config and setup
env_conf = {
    'set_on_rack': False,
    'render': False,
    'model_name': 'default_humanoid_mocap',
    'mocap': False
}

env = BipedEnv(**env_conf)



N = 1
s_opts = {"max_iter": 5}
act_dim = env.sim.data.ctrl.shape[0]
q0 = np.zeros(env.sim.data.qpos.shape[0])
q0[2] = 5

cas_model = mujoco_model(                    
                            env = env,
                            N = N,
                            q0 = q0,
                            dq0= np.zeros(env.sim.data.qvel.shape[0]),
                            u0 = np.zeros(env.sim.data.ctrl.shape[0]),
                        )



dt = env.model.opt.timestep
opti = cas.Opti()

u = opti.variable(act_dim*N)
opti.minimize(  cas_model(u)   )

opti.solver('ipopt',{},s_opts)


# print(opti.variable)
# exit()
opti.solve()

# print(sol.value(u))



exit()
#####
T = 10. # Time horizon
N = 20 # number of control intervals

# Declare model variables
x1 = MX.sym('x1')
x2 = MX.sym('x2')
x = vertcat(x1, x2)
u = MX.sym('u')

# q = MX.sym('q',env.sim.data.qpos.shape[0])
# dq = MX.sym('dq',env.sim.data.qvel.shape[0])
# x = vertcat(q,dq)
# u = MX.sym('u',env.sim.data.ctrl.shape[0])



# Model equations
xdot = vertcat((1-x2**2)*x1 - x2 + u, x1)


#cas_model
print(xdot)


# xdot = 
# Objective term
L =  u**2

# Formulate discrete time dynamics
if False:
   # CVODES from the SUNDIALS suite
   dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
   opts = {'tf':T/N}
   F = integrator('F', 'cvodes', dae, opts)
else:
   # Fixed step Runge-Kutta 4 integrator
   M = 4 # RK4 steps per interval
   DT = T/N/M
   #print(DT,T,N,M)
   #exit()
   f = Function('f', [x, u], [xdot, L])
   X0 = MX.sym('X0', 2)
   U = MX.sym('U')
   X = X0
   Q = 0 #Why?
   for sj in range(M):
       k1, k1_q = f(X, U)
       print(sj,k1,k1_q)
       k2, k2_q = f(X + DT/2 * k1, U)
       k3, k3_q = f(X + DT/2 * k2, U)
       k4, k4_q = f(X + DT * k3, U)
       X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
       Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
   F = Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])
   print(F)

exit()
# Evaluate at a test point
Fk = F(x0=[0.2,0.3],p=0.4)
print(Fk['xf'])
print(Fk['qf'])

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

# Formulate the NLP
Xk = MX([0, 1])
for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym('U_' + str(k))
    w += [Uk]
    lbw += [-1]
    ubw += [1]
    w0 += [0]

    # Integrate till the end of the interval
    Fk = F(x0=Xk, p=Uk)
    Xk = Fk['xf']
    J=J+Fk['qf']

    # Add inequality constraint
    g += [Xk[0]]
    lbg += [-.25]
    ubg += [inf]


# Create an NLP solver
prob = {'f': J, 
'x': vertcat(*w), 
'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)

# Solve the NLP
sol = solver(
                x0=w0, 
                lbx=lbw, 
                ubx=ubw, 
                lbg=lbg, 
                ubg=ubg)
w_opt = sol['x']

# Plot the solution
u_opt = w_opt
x_opt = [[0, 1]]
for k in range(N):
    Fk = F(x0=x_opt[-1], p=u_opt[k])
    x_opt += [Fk['xf'].full()]
x1_opt = [r[0] for r in x_opt]
x2_opt = [r[1] for r in x_opt]

tgrid = [T/N*k for k in range(N+1)]
import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
plt.plot(tgrid, x1_opt, '--')
plt.plot(tgrid, x2_opt, '-')
plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
plt.xlabel('t')
plt.legend(['x1','x2','u'])
plt.grid()
plt.show()