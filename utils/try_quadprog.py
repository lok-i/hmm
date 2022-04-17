



from numpy import array, dot
from qpsolvers import solve_qp
import numpy as np






ik_soln_filpath = './data/our_data/ik_solns/AB3_Session1_Right10_Left10_from_2000_to_2500.npz'  
id_soln_filpath = './data/our_data/id_solns/AB3_Session1_Right10_Left10_from_2000_to_2500.npz'  

ik_solns = np.load(ik_soln_filpath)['ik_solns']
id_solns = np.load(id_soln_filpath)['id_solns']

print(ik_solns.shape,id_solns.shape)

# exit()

control_length = 21
variable_size = ik_solns.shape[1] + id_solns.shape[1] + control_length

print(variable_size)
# exit()
M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])

P = dot(M.T, M)  # this is a positive definite matrix
P = np.eye(N=variable_size)#dot(M.T, M)  # this is a positive definite matrix

q = dot(array([3., 2., 3.]), M)
q = np.zeros(variable_size)

G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
G = np.random.random(size=(variable_size,variable_size)) #array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])

h = array([3., 2., -2.])
h = np.full(variable_size,0)

A = array([1., 1., 1.])
A = np.diag( np.concatenate(
                                [
                                    np.zeros(variable_size - control_length),
                                    np.ones(6),
                                    np.zeros(control_length-6)
                                ]
                            ).ravel()
            )                        #np.eye(variable_size) #array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])


b = array([1.])
b = np.zeros(variable_size)


print(G.shape,h.shape,A.shape,b.shape)

x = solve_qp(P=P, 
                q=q, 
                # G, h, 
            # A, b
                )
print("QP solution: x = {}".format(x))