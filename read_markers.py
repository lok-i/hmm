import c3d
import numpy as np
import matplotlib.pyplot as plt
import yaml
import matplotlib.pyplot as plt
from pytransform3d.transformations import   random_transform,transform_from, plot_transform
from pytransform3d.rotations import random_axis_angle, matrix_from_axis_angle
from pytransform3d.plot_utils import make_3d_axis, plot_capsule



random_state = np.random.RandomState(42)
# ax = make_3d_axis(1)

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


time_step = 0
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

# print(pts_in_link.shape)
# exit()





ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()
ax.view_init(elev=0, azim=-90)

capsule2origin = np.eye(4)#random_transform(random_state)
for i in range(3):
    capsule2origin[i][3] = np.mean(pts_in_link[:,i])

print(capsule2origin)

height = 0.2
radius = 0.05
plot_transform(ax=ax, A2B=capsule2origin, s=0.03)
plot_capsule(ax=ax, A2B=capsule2origin, height=height, radius=radius,
             color="r", alpha=0.5, wireframe=False)

plt.show()


