from math import tau
from tkinter import Y
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import misc_functions

q_diff = np.load('./data/q_diff.npy')
# convert quat to rpy in qpos
q_diff = np.array([ base_pos.tolist() + misc_functions.quat2euler(base_quat).tolist() + joint_pos.tolist() \
            for base_pos, base_quat, joint_pos in 
            zip( q_diff[:,0:3], q_diff[:,3:7], q_diff[:,7:] ) ] )      

dq_diff = np.load('./data/dq_diff.npy')
tau_diff = np.load('./data/tau_diff.npy')
print(q_diff.shape,dq_diff.shape,tau_diff.shape)


X = np.array([ delta_q.tolist()+delta_dq.tolist() for delta_q,delta_dq in zip(q_diff,dq_diff) ])#np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
print('X \'s Shape:',X.shape)
# y = 1 * x_0 + 2 * x_1 + 3
y = tau_diff#np.dot(X, np.array([1, 2])) + 3

nan_chk = np.isnan(y) 

for i in range(nan_chk.shape[0]):
    for j in range(nan_chk.shape[1]):
        if nan_chk[i,j]:
            print(i,j)
            y[i,j] = 1.
            y[i,j] = 0.5*(y[i-1,j] + y[i+1,j] )

# exit()

print("Y \'s shape", y.shape)
reg = LinearRegression().fit(X, y)
print("Regression Score:",reg.score(X, y))

# print( np.min(reg.coef_) )
fig,ax = plt.subplots(1,2,gridspec_kw={'width_ratios': [5, 1]})

print("Linear Model fit as, Y = Ax + B")
print( 'A\'s shape:',reg.coef_.shape)

tau_labels = [
                'base_x','base_y','base_z',
                'base_ro','base_pi','base_yw',

                'abdmn_x','abdmn_y','abdmn_z',
                'lhip_x','lhip_z','lhip_y', 'lknee','lankle_y','lankle_x',
                'rhip_x','rhip_z','rhip_y', 'rknee','rankle_y','rankle_x',
                ''
]
state_labels = [
                
                'base_x','base_y','base_z',
                'base_ro','base_pi','base_yw',
                'abdmn_x','abdmn_y','abdmn_z',
                'lhip_x','lhip_z','lhip_y', 'lknee','lankle_y','lankle_x',
                'rhip_x','rhip_z','rhip_y', 'rknee','rankle_y','rankle_x',

                'v_base_x','v_base_y','v_base_z',
                'v_base_ro','v_base_pi','v_base_yw',
                'v_abdmn_x','v_abdmn_y','v_abdmn_z',
                'v_lhip_x','v_lhip_z','v_lhip_y', 'v_lknee','v_lankle_y','v_lankle_x',
                'v_rhip_x','v_rhip_z','v_rhip_y', 'v_rknee','v_rankle_y','v_rankle_x',
                
                ''
]

#####

im1 = ax[0].imshow( reg.coef_,cmap='seismic')
ax[0].set_title("A")

ax[0].set_yticks(np.arange(-0.5, reg.coef_.shape[0]))
ax[0].set_yticklabels(tau_labels,verticalalignment="top")

xlabels = [ label for label in np.arange(0,reg.coef_.shape[1]) ]
xlabels.append('')
ax[0].set_xticks(np.arange(-0.5, reg.coef_.shape[1]))
ax[0].set_xticklabels(state_labels,horizontalalignment="left",rotation=90)
ax[0].grid()

############
ax[1].set_title("B")
intercept_as_2d = np.atleast_2d( reg.intercept_ ).T
print( 'B\'s shape:',intercept_as_2d.shape)
im2 = ax[1].imshow( intercept_as_2d ,cmap='seismic')


        
ax[1].set_yticks(np.arange(-0.5, intercept_as_2d.shape[0]))
ax[1].set_yticklabels(tau_labels ,verticalalignment="top")

xlabels = [ label for label in np.arange(0,intercept_as_2d.shape[1]) ]
xlabels.append('')
ax[1].set_xticks(np.arange(-0.5, intercept_as_2d.shape[1]))
ax[1].set_xticklabels(xlabels,horizontalalignment="left",rotation=-45)

ax[1].grid()


fig.colorbar(im1, ax=ax[0])
fig.colorbar(im2, ax=ax[1])
fig.suptitle('Linear Model, Y = AX+B')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax[1])
# cax = divider.append_axes("right", size="50%", pad=0.1)

plt.show()