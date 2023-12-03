import math
import numpy as np
from scipy.optimize import minimize
import numdifftools as nd


def steepest_descent(func, xyz, precision):
    f_old = math.inf
    f_new = func(xyz)

    while precision < abs(f_old - f_new):
        grad_x, grad_y, grad_z = nd.Gradient(func)(xyz)

        gamma_values = np.logspace(np.log10(0.0001), np.log10(1), num=1000)
        function_values = [func([xyz[0] - gamma * grad_x, xyz[1] - gamma * grad_y, xyz[2] - gamma * grad_z]) for gamma in gamma_values]
        min = math.inf
        step_size = 0
        for i in range(1, len(function_values)):
            if min > function_values[i]:
                min = function_values[i]
                step_size = gamma_values[i]
            if function_values[i - 1] < function_values[i]:
                break

        xyz[0] -= step_size * grad_x
        xyz[1] -= step_size * grad_y
        xyz[2] -= step_size * grad_z
        f_old = f_new
        f_new = func(xyz)

    return xyz


f = lambda xyz: -(xyz[0] * xyz[1] * xyz[2])
h0 = lambda xyz: 1 - xyz[0] - xyz[1] - xyz[2]
g0 = lambda xyz: xyz[0]
g1 = lambda xyz: xyz[1]
g2 = lambda xyz: xyz[2]

xyz0 = [0, 0, 0]
xyz1 = [1, 1, 1]
xyz2 = [0.6, 0, 0.9]

xyz = xyz0
r = 0.5

f_old = math.inf
f_new = f(xyz)

while 10 ** -6 < abs(f_old - f_new):
    print(xyz, f(xyz))
    curr_func = lambda xyz: f(xyz) + r * ((min(0, g0(xyz)) ** 2 + min(0, g1(xyz)) ** 2 + min(0, g2(xyz)) ** 2) + h0(xyz) ** 2)
    xyz = steepest_descent(curr_func, xyz, 10 ** -4)
    r *= 1.2
    f_old = f_new
    f_new = curr_func(xyz)
print(xyz, f(xyz))

