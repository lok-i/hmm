import pybullet as p
import time
import pybullet_data


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1.5]
startOrientation = p.getQuaternionFromEuler([0,0,0])
mujoco_body = p.loadURDF("result/humanoid_torso.urdf",startPos,startOrientation)


maxForce = 0
mode = p.VELOCITY_CONTROL
for jointIndex in range(p.getNumJoints(mujoco_body)):
	p.setJointMotorControl2(mujoco_body, jointIndex,controlMode=mode, force=maxForce)


#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(mujoco_body, startPos, startOrientation)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
