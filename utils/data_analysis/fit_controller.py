from math import tau
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

q_diff = np.load('./data/q_diff.npy')
dq_diff = np.load('./data/dq_diff.npy')
tau_diff = np.load('./data/tau_diff.npy')
print(q_diff.shape,dq_diff.shape,tau_diff.shape)


X = q_diff#np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
print(np.max(X))
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

# print(y.shape)
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))
# print(reg.coef_.shape)
# print(reg.intercept_.shape)


# print( np.min(reg.coef_) )
ax = plt.subplot()
# im = ax.imshow( np.clip(reg.coef_,-1000,1000))

im = ax.imshow(X[0:100,:])

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.show()