import numpy as np
import random

import matplotlib.pyplot as plt

###################################################
#### MAS374 Programming Homework Assignemnt #3 ####
####                       20180127 Woojin Kim ####
###################################################

#### ---- code for part (a) ---- ####

def my_lstsq(A, y):
    """
        Fill in your code here. (Erase this comment if you want.)
        Read the description, and write a function that does the required.
    """
    U, S, Vh = np.linalg.svd(A, full_matrices=False) # To get U_r, I disabled full_matrices paramter.
    S_mat = np.diag(S)

    temporal_result = np.matmul(np.transpose(Vh), np.linalg.inv(S_mat))
    A_dagger = np.matmul(temporal_result, np.transpose(U))
    theta   = np.matmul(A_dagger, y)

    return theta


#### ---- code for part (b) ---- ####

y_b = np.zeros(250)       # initialize vector of labels

"""
    Fill in your code here. (Erase this comment if you want.)
    Here, you should:
    (1) construct an appropriate matrix /A/ by sampling 250 random points,
    (2) fill in the vector of labels /y/ appropriately, and
    (3) find the optimal solution /theta/ using /my_lstsq/.
"""

# For (1) constructing an appropriate matrix /A/ by sampling 250 random points
# Generate 250 random samples unformly distributed from [-2, 2] * [-2, 2]
samples_b = 4 * np.random.random_sample((250, 2)) - 2

# For (2) filling in the vector of lables /y/ appropriately,
# label the sample points
for i in range(250):
    [x1, x2] = samples_b[i]
    
    if ((x1 ** 2 + x2 ** 2) <= 1):
        y_b[i] = 1
    else:
        y_b[i] = -1

# For (3) finding the optimal solution /theta/ using /my_lstsp/,
# convert the problem and use my_lstsq

# Each row of A should contain the elements of the form (1, x1, x2, x1^2, x1x2, x2^2)
# where (x1, x2) is one of the samples. 
A_b = np.zeros((250, 6));

for i in range(250):
    [x1, x2] = samples_b[i] # ith sample
    
    A_b[i][0] = 1
    A_b[i][1] = x1
    A_b[i][2] = x2
    A_b[i][3] = x1 ** 2
    A_b[i][4] = x1 * x2
    A_b[i][5] = x2 ** 2

# Get the optimal solution for A/theta/ = y
theta_b = my_lstsq(A_b, y_b)

print("The theta value for problem (b) is")
print(theta_b)  # you may change the name of the variable if you want.

# The function for computing predicted label value 
# under give theta and sample point
def f(theta_, x1_, x2_):
    return theta_[0] + theta_[1] * x1_ + theta_[2] * x2_ + theta_[3] * (x1_ ** 2) + theta_[4] * x1_ * x2_ + theta_[5] * (x2_ ** 2)

x_axis = [element[0] for element in samples_b]
y_axis = [element[1] for element in samples_b]
meshgrid_x, meshgrid_y = np.meshgrid(x_axis, y_axis)
result = f(theta_b, meshgrid_x, meshgrid_y)

plt.title("contour plot for (b)")

#CS = plt.contour(meshgrid_x, meshgrid_y, result, colors="black")
CS = plt.contourf(meshgrid_x, meshgrid_y, result, alpha=0.75, cmap='jet')
CB = plt.colorbar(CS)
plt.show()

#### ---- code for part (c) ---- ####

y_c = np.zeros(250)       # initialize vector of labels 

"""
    Fill in your code here. (Erase this comment if you want.)
    Here, you should:
    (1) construct an appropriate matrix /A/ by sampling 250 random points,
    (2) fill in the vector of labels /y/ appropriately, and
    (3) find the optimal solution /theta/ using /my_lstsq/.
"""

# For (1) constructing an appropriate matrix /A/ by sampling 250 random points
# Generate 250 random samples unformly distributed from [0, 2] * [0, 2]
samples_c = 2 * np.random.random_sample((250, 2))

# For (2) filling in the vector of lables /y/ appropriately,
# label the sample points
for i in range(250):
    [x1, x2] = samples_c[i]
    
    if ((x1 ** 2 + x2 ** 2) <= 1):
        y_c[i] = 1
    else:
        y_c[i] = -1

# For (3) finding the optimal solution /theta/ using /my_lstsp/,
# convert the problem and use my_lstsq

# Each row of A should contain the elements of the form (1, x1, x2, x1^2, x1x2, x2^2)
# where (x1, x2) is one of the samples. 
A_c = np.zeros((250, 6));

for i in range(250):
    [x1, x2] = samples_c[i] # ith sample
    
    A_c[i][0] = 1
    A_c[i][1] = x1
    A_c[i][2] = x2
    A_c[i][3] = x1 ** 2
    A_c[i][4] = x1 * x2
    A_c[i][5] = x2 ** 2

# Get the optimal solution for A/theta/ = y
theta_c = my_lstsq(A_c, y_c)

print("")
print("The theta value for problem (c) is")
print(theta_c)  # you may change the name of the variable if you want.

x_axis = [element[0] for element in samples_c]
y_axis = [element[1] for element in samples_c]
meshgrid_x, meshgrid_y = np.meshgrid(x_axis, y_axis)
result = f(theta_c, meshgrid_x, meshgrid_y)

plt.title("contour plot for (c)")

#CS = plt.contour(meshgrid_x, meshgrid_y, result, colors="black")
CS = plt.contourf(meshgrid_x, meshgrid_y, result, alpha=0.75, cmap='jet')
CB = plt.colorbar(CS)
plt.show()