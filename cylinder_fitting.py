import numpy as np
import c3d
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import yaml
from pytransform3d.transformations import   random_transform,transform_from, plot_transform
from pytransform3d.rotations import random_axis_angle, matrix_from_axis_angle
from pytransform3d.plot_utils import make_3d_axis, plot_capsule

def error_func(params,x,y,z):
    '''
    Args:
    params: the params of the optimsiation
    params[0] -> x co ordinate of centre point C
    params[1] -> y co ordinate of centre point C
    params[2] -> z co ordinate of centre point C
    params[3] -> ith element of basis W
    params[4] -> jth element of basis W
    params[5] -> kth element of basis W
    params[6] -> r**2, radius squared

    x,y,z -> co-ordinates of th data point

    '''
    I = np.identity(3)
    
    C = np.atleast_2d(np.array([params[0],params[1],params[2]]))
    W = np.array([params[3],params[4],params[5]])
    WWT = np.dot(np.atleast_2d(W).T, np.atleast_2d(W))
    costs = []
    for xi,yi,zi in zip(x,y,z):
        X = np.atleast_2d(np.array([xi,yi,zi]))
        # print(x,y,z)
        error = np.linalg.multi_dot([(X-C),(I-WWT),(X-C).T]) - params[6]**2
        costs.append(error[0][0])
    # print(costs)
    return costs



assets_path = './gym_hmm_ec/envs/assets/our_data/'
marker_conf_file_name = 'marker_config.yaml'
c3d_file_name = 'mocap_data/Trial_1.c3d'


config_file = open(assets_path + marker_conf_file_name,'r+')
marker_conf = yaml.load(config_file, Loader=yaml.FullLoader)

link_marker_info = marker_conf['segment2markers']
 
pts_to_plt = []
with open( assets_path+c3d_file_name , 'rb') as handle:

    reader = c3d.Reader(handle)
    for data in reader.read_frames():

        # print('Frame {}'.format(data[0],data[1].shape,data[2][0]))
        all_marker = []
        
        if data[0] % 100 == 0:
            for pt in data[1]:
                all_marker.append(pt[0:3].tolist())
        
            pts_to_plt.append(all_marker)
        
        # if data[0] > 100:
        #     break
pts_to_plt = np.array(pts_to_plt)


time_step = 100
marker_pos_at_time_step = pts_to_plt[time_step]
# print(marker_pos_at_time_step.shape)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')


link_names = ['right_hip','right_upper_leg','right_knee']

pts_in_link = []

for link_name in link_names:
    marker_index_of_link = list(link_marker_info[link_name].keys())
    for marker_index in marker_index_of_link:

        marker_pos_in_m = 0.001* marker_pos_at_time_step[marker_index,0:3] # mm -> m
        
        pts_in_link.append(marker_pos_in_m.tolist())
        marker_name = link_marker_info[link_name][marker_index]
        ax.scatter(marker_pos_in_m[0],marker_pos_in_m[1],marker_pos_in_m[2],label=marker_name)

        
pts_in_link = np.array(pts_in_link)
print(pts_in_link.shape)
# print(pts_in_link)

x = pts_in_link[:,0]
y = pts_in_link[:,1]
z = pts_in_link[:,2]

# x = np.array([0,0,-2,2,1.414,-1.414,1.414])
# y = np.array([-2,2,0,0,1.414,-1.414,-1.414])
# z = np.array([0,0,0,0,0,0,0])

# x = np.random.uniform(low=0,high=1,size=7)
# y = np.random.uniform(low=0,high=1,size=7)
# z = np.random.uniform(low=0,high=1,size=7)


params0 = [0,0.1,-0.5,0.1,0.5,2,5]
est_p , success = leastsq(error_func, params0, args=(x, y, z), maxfev=1000)
# print(leastsq(error_func, params0, args=(x, y, z), maxfev=1000,full_output=True))
print(est_p,success)



capsule2origin = np.eye(4)#random_transform(random_state)

# let 1,1,k be a vector perpendicular to W
if est_p[3] != 0:
    k = (-1* est_p[5]-1* est_p[4] ) / est_p[3]
    x_cap = np.array([k,1,1] )

elif est_p[4] != 0:
    k = (-1* est_p[3]-1* est_p[5] ) / est_p[4] 
    x_cap = np.array([1,k,1] )
 
else:
    k = (-1* est_p[3]-1* est_p[4] ) / est_p[5] 
    x_cap = np.array([1,1,k] )


capsule2origin[:3,0] = (1.0/np.linalg.norm(x_cap))*x_cap
y_cap = np.cross(est_p[3:6],x_cap)
capsule2origin[:3,1] = (1.0/np.linalg.norm(y_cap))*y_cap
capsule2origin[:3,2] = (1.0/np.linalg.norm(est_p[3:6]))*est_p[3:6]
capsule2origin[:3,3] = est_p[0:3]

# print(capsule2origin)
# print(np.linalg.det(capsule2origin[0:3,0:3]))
height = 2
radius = np.abs(est_p[6])
plot_transform(ax=ax, A2B=capsule2origin, s=0.03)
plot_capsule(ax=ax, A2B=capsule2origin, height=height, radius=radius,
             color="r", alpha=0.5, wireframe=False)

ax.scatter(est_p[0],est_p[1],est_p[2],label='C')
pt2 = est_p[0:3] + est_p[3:6]

ax.plot( [est_p[0],pt2[0]],
         [est_p[1],pt2[1]],
         [est_p[2],pt2[2]],
         label='W')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()
ax.view_init(elev=0, azim=-90)
plt.show()
