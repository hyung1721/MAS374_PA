### 20180127 Kim Woojin 
### MAS374 Homework #9

import numpy as np

### Helper function ###

def l2_norm_of_change(x, y) :
    """
        Compute the l2-norm of the change in x and y, ||x-y||_2
    """
    n = len(x)
    result = 0

    for i in range(n):
        result = result + (x[i] - y[i])**2
    
    return result ** 0.5

#### ---- Problem 2(a) ---- ####

def dual_proj(l):
    """
        Fill in your code here. (Erase this comment if you want.)
        Read the description, and write a function that does the required.
    """
    n = len(l)
    proj_l = [0] * n

    for i in range(n):
        if l[i] < 0:
            proj_l[i] = 0
        else:
            proj_l[i] = l[i]

    return proj_l

#### ---- Problem 2(b) ---- ####
    
def dual_grad(l, x, A, b):
    """
        Fill in your code here. (Erase this comment if you want.)
        Read the description, and write a function that does the required.
    """
    # We know that the gradient of the objective dunction in (4) at l is
    # A*A^t*l + b - A*x

    first_term  = (A @ A.T) @ l
    second_term = b - A @ x
    return first_term + second_term

#### ---- Problem 2(c) ---- ####
    
def solve_dual(x, A, b):
    tol = 2**-40
    """
        Fill in your code here. (Erase this comment if you want.)
        Read the description, and write a function that does the required.
    """
    n = len(b)

    # initial lambda
    l = [0] * n 
    # For accuracy
    difference = 9999
    # Choose constant stepsize as described in report P2-(c)
    step_size = 1 / np.linalg.norm(A @ A.T, 'fro')
    while difference >= tol:
        l_temp = l - step_size * dual_grad(l, x, A, b)
        l_next = dual_proj(l_temp)
        
        # Compute how much [x]_X is changed
        proj_x      = x - A.T @ l
        proj_x_next = x - A.T @ l_next
        difference = l2_norm_of_change(proj_x_next, proj_x)

        # Update
        l = l_next

    return l
    
#### ---- Problem 3(a) ---- ####
   
def prim_proj(x, A, b):
    """
        Fill in your code here. (Erase this comment if you want.)
        Read the description, and write a function that does the required.
    """
    # As the solution of Problem 1 is a result of projection of x onto X,
    # we can use the above function, solve_dual() to compute [x]_X
    # projection_x = x - A^t * optimal_lambda -> we can compute optimal lambda using solve_dual()

    optimal_lamda = solve_dual(x, A, b)
    proj_x = x - A.T @ optimal_lamda
    
    return proj_x

#### ---- Problem 3(b) ---- ####

def grad_f0(x, H, c) :
    """
        Fill in your code here. (Erase this comment if you want.)
        Read the description, and write a function that does the required.
    """
    return H @ x + c
    
def f0(x, H, c) :
    """
        Fill in your code here. (Erase this comment if you want.)
        Read the description, and write a function that does the required.
    """
    return 0.5 * (x.T @ H) @ x + c.T @ x
    
#### --  A helper function which prints the results in a given format -- ####

def print_results(x_opt, H, c):
    np.set_printoptions(floatmode="unique") # print with full precision 
    print("optimal value p* =")
    print("", f0(x_opt, H, c), sep = "\t")
    print("\noptimal solution x* =")

    for coord in x_opt :
        print("", coord, sep = '\t')
    
    return

# first example in page 3 of the document, 
# written for you so you can test your code. 

# First example
"""
H = np.array([[6,  4],
              [4, 14]])
c = np.array([-1, -19])

A = np.array([[-3,  2],
              [-2, -1],
              [ 1,  0]])
b = np.array([-2, 0, 4])
"""

# Second example
"""
H = np.array([[20, -5,  3],
              [-5, 17,  0],
              [ 3,  0, 10]])

c = np.array([4, 2, 7])

A = np.array([[-1,  0,  0],
              [ 0, -1,  0],
              [ 0,  0, -1],
              [ 1,  1,  1],
              [-1, -1, -1]])
b = np.array([0, 0, 0, 1, -1])
"""

# Third example
#"""
H = np.array([[34,   4,  15,  10,   2],
              [ 4,  35, -17,   3,  -8],
              [15, -17,  36, -12,  -4],
              [10,   3, -12,  37, -11],
              [ 2,  -8,  -4, -11,  38]])
c = np.array([-1, 17, -2, 9, 0])

A = np.array([[ 10, 18, 12,   2,   0],
              [ 18, -1,  1,  19,  -2],
              [  6, -7, 18,   8,  -6],
              [-13, 19, 11, -10,  -2],
              [  3,  5, -1,  -4, -13],
              [ -4,  0, 11,  19,  -8]])
b = np.array([16, 7, 10, 4, 17, 18])
#"""

#### ---- Problem 3(c) ---- ####

eps = 2**-40
        
"""
    Fill in your code here. (Erase this comment if you want.)
    Read the description, and write a function that does the required.
"""

n = len(c)

# initial x
x_opt = [0] * n 
# For accuracy
difference = 9999
# Choose constant stepsize as described in report P3-(c)
# that is, the inverse of frobenuis norm of H
step_size = 1 / np.linalg.norm(H, 'fro')

while difference >= eps:
    x_temp = x_opt - step_size * grad_f0(x_opt, H, c)
    x_next = prim_proj(x_temp, A, b)
    difference = l2_norm_of_change(x_next, x_opt)

    # Update
    x_opt = x_next

# printing the results
print_results(x_opt, H, c)
