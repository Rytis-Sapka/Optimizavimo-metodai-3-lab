from Methods import *

def createOriginalFunction(r):
    def new_function(args):
        x = args[0]
        y = args[1]
        z = args[2]
        return -(x*y*z) + (1 / r) * (max(0, -x+0.001) ** 2 + max(0, -y+0.001) ** 2 + max(0, -z+0.001) ** 2 + (2*x*y + 2*x*z + 2*y*z - 1) ** 2)
    return new_function

start = [0.6, 0.3, 0]
r = 1
steps = []

for i in range(1000):
    steps.append(start)
    _, result = fastest_descent_3d(start, createOriginalFunction(r))
    print(result)
    print(createOriginalFunction(r)(result))
    print(r)
    if (abs(result[0] - start[0]) + abs(result[1] - start[1]) + abs(result[2] - start[2]) < 0.001):
        break
    start = result
    r = r * 0.9

graph_3d(steps)
print(len(steps))