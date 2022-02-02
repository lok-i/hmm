import numpy as np
import c3d
from scipy.spatial import ConvexHull
from scipy.optimize import leastsq,least_squares
import matplotlib.pyplot as plt
import yaml
from pytransform3d.transformations import   random_transform,transform_from, plot_transform
from pytransform3d.rotations import random_axis_angle, matrix_from_axis_angle
from pytransform3d.plot_utils import make_3d_axis, plot_capsule

# load

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
    W = np.atleast_2d(np.array([params[3],params[4],params[5]]))
    WWT = np.dot(np.atleast_2d(W).T, np.atleast_2d(W))
    h = 0.4 # temp
    costs = []
    for xi,yi,zi in zip(x,y,z):
        X = np.atleast_2d(np.array([xi,yi,zi]))
        first_error = np.linalg.multi_dot([(X-C),(I-WWT),(X-C).T]) - params[6]**2
        second_error = (W.dot((X-C).T))**2 - 0.25*(h**2)
        total_error = first_error[0][0] + second_error[0][0]
        costs.append(total_error)
    # print(costs)
    return costs


time_step = 0
marker_pos_at_time_step = pts_to_plt[time_step]
# print(marker_pos_at_time_step.shape)

# intialise plot and scatter points

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


link_names = [ 
                # 'left_hip','left_upper_leg','left_knee',
                'right_hip','right_upper_leg','right_knee'
            ]

pts_in_link = []

for link_name in link_names:
    marker_index_of_link = list(link_marker_info[link_name].keys())
    for marker_index in marker_index_of_link:

        marker_pos_in_m = 0.001* marker_pos_at_time_step[marker_index,0:3] # mm -> m
        
        pts_in_link.append(marker_pos_in_m.tolist())
        marker_name = link_marker_info[link_name][marker_index]
        ax.scatter(marker_pos_in_m[0],marker_pos_in_m[1],marker_pos_in_m[2],label=marker_name)

        
pts_in_link = np.array(pts_in_link)
print("No. of points:",pts_in_link.shape)


x = pts_in_link[:,0]
y = pts_in_link[:,1]
z = pts_in_link[:,2]

hull = ConvexHull(pts_in_link)

# x = np.array([0,0,-2,2,1.414,-1.414,1.414])
# y = np.array([-2,2,0,0,1.414,-1.414,-1.414])
# z = np.array([0,0,0,0,0,0,0])

# x = np.random.uniform(low=0,high=1,size=7)
# y = np.random.uniform(low=0,high=1,size=7)
# z = np.random.uniform(low=0,high=1,size=7)


min_bound = [np.min(x),np.min(y),np.min(z),-np.inf,-np.inf,-np.inf, 0]
max_bound = [np.max(x),np.max(y),np.max(z), np.inf, np.inf, np.inf, np.inf]


mid_knee = 0.5* (pts_in_link[-1] + pts_in_link[-2] )
hip_pt = pts_in_link[0]
C0 = 0.5 * ( hip_pt+ mid_knee ) 
W0 = np.array([0,0,1.])
r0 = 0.1 # in meteres

params0 =  np.concatenate((C0,W0,[r0]))    #[]#0.5*(np.array(max_bound) + np.array(min_bound) ) #[0,0,-0.5,0.1,0.5,2,5]

print("rough length of the link:", np.linalg.norm(hip_pt-mid_knee))
# for _min, _max in zip(min_bound,max_bound):
#     if _min != -np.inf and _max != np.inf:
#         params0.append( 0.5*(_min +_max))
#     else:
#         params0.append(0)

print("Initial Parameters",params0)

# do least squares
res_1 = least_squares(  error_func, 
                        params0, 
                        args=(x, y, z),
                        bounds=( 
                                min_bound,
                                max_bound                               
                                )
                        
                        )
print(" Sollution Parameters",res_1.x)



#### other plotts

capsule2origin = np.eye(4)#random_transform(random_state)


# C = est_p[0:3]
# W = est_p[3:6]
# R = np.abs(est_p[6])

C = res_1.x[0:3]
W = res_1.x[3:6]
R = np.abs(res_1.x[6])
# H2 = 0
# for pt in pts_in_link:
#     H2 += (W.dot((pt-C).T))**2
#     print("Height:",pt,0.5*np.sqrt(H2) )
     

# print("Height:",0.5* (np.sqrt(H2)/pt.shape[0]) )


# let 1,1,k be a vector perpendicular to W

print(C,W,R)
if W[1] != 0:
    k = (-1*W[1]-1*W[2] ) /W[1]
    x_cap = np.array([k,1,1] )

elif W[2] != 0:
    k = (-1*W[1]-1*W[1] ) /W[2] 
    x_cap = np.array([1,k,1] )
 
else:
    k = (-1*W[1]-1*W[2] ) /W[1] 
    x_cap = np.array([1,1,k])


capsule2origin[:3,0] = (1.0/np.linalg.norm(x_cap))*x_cap
y_cap = np.cross(W,x_cap)
capsule2origin[:3,1] = (1.0/np.linalg.norm(y_cap))*y_cap
capsule2origin[:3,2] = (1.0/np.linalg.norm(W))*W
capsule2origin[:3,3] = C

# print(capsule2origin)
# print(np.linalg.det(capsule2origin[0:3,0:3]))
height = 2
radius = R
plot_transform(ax=ax, A2B=capsule2origin, s=0.03,strict_check=False)
# plot_capsule(ax=ax, A2B=capsule2origin, height=height, radius=radius,
#              color="r", alpha=0.5, wireframe=False)

for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    ax.plot(pts_in_link[s, 0], pts_in_link[s, 1], pts_in_link[s, 2], "r-")


ax.scatter(C0[0],C0[1],C0[2],label='C0')
ax.scatter(C[0],C[1],C[2],label='C')


ax.plot( [C0[0],C0[0]+W0[0]],
         [C0[1],C0[1]+W0[1]],
         [C0[2],C0[2]+W0[2]],
         label='W0')
ax.plot( [C[0],C[0]+W[0]],
         [C[1],C[1]+W[1]],
         [C[2],C[2]+W[2]],
         label='W')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim([-0.25,0.25])
ax.set_ylim([0.5,1.0])
ax.set_zlim([0.5,1.0])
ax.legend()
ax.view_init(elev=0, azim=90)
plt.show()
