

# basic ID
from __future__ import print_function
 
import pinocchio as pin
import pinocchio as se3
from quadprog import solve_qp
import numpy as np 
import sys
from os.path import dirname, join, abspath
from pinocchio.visualize import MeshcatVisualizer

# urdf_model_path = 'result/simple_humanoid.urdf' 
urdf_model_path = './urdfs/humanoid_torso.urdf' 
mesh_dir = ''

model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())
data = model.createData()

# q = pin.neutral(model)
# v = pin.utils.zero(model.nv)
# a = pin.utils.zero(model.nv)

# q[7] = np.radians(10)
# print(q.shape)
# print(v.shape)
# print(a.shape)


q =  pin.neutral(model)
vq = pin.utils.rand(model.nv)
aq0 = pin.utils.zero(model.nv)
# compute dynamic drift -- Coriolis, centrifugal, gravity
b = se3.rnea(model, data, q, vq, aq0)
# compute mass matrix M
M = se3.crba(model, data, q)

print(M.shape, b.shape,aq0.shape)
# exit()
tau_1 = pin.rnea(model,data,q,vq,aq0)
tau_2 = M.dot(aq0) + b

print('tau pin = ', tau_1.T,'\n',tau_1.shape)
print('tau se3 = ', tau_2.T,'\n',tau_2.shape)




'''
# Load the URDF model.
# Conversion with str seems to be necessary when executing this file with ipython
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))),"models")


model_path = join(pinocchio_model_dir,"example-robot-data/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "humanoid_torso.urdf"
urdf_model_path = "./urdfs/" +urdf_filename #join(join(model_path,"talos_data/robots"),urdf_filename)

print(urdf_model_path)
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir, pin.JointModelFreeFlyer())
viz = MeshcatVisualizer(model, collision_model, visual_model)

try:
    viz.initViewer(open=True)
except ImportError as err:
    print("Error while initializing the viewer. It seems you should install Python meshcat")
    print(err)
    sys.exit(0)

# Load the robot in the viewer.
viz.loadViewerModel()

# Display a robot configuration.
q0 = pin.neutral(model)
viz.display(q0)
viz.displayCollisions(True)
viz.displayVisuals(False)
'''
