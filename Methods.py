import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd

MAX_ITERATIONS = 1000
TOLERANCE = 0.0001

def gradient_descent_3d(start, learn_rate, originalFunc):
    x = start[0]
    y = start[1]
    z = start[2]
    steps = [start]

    for _ in range(MAX_ITERATIONS):
        gradientX, gradientY, gradientZ = nd.Gradient(originalFunc)([x, y, z])
        diffX = learn_rate * gradientX
        diffY = learn_rate * gradientY
        diffZ = learn_rate * gradientZ
        if np.abs(diffX) + np.abs(diffY) + np.abs(diffZ) < TOLERANCE:
            break
        x -= diffX
        y -= diffY
        z -= diffZ
        steps.append((x, y, z)) 
  
    return steps, [x, y, z]

def fastest_descent_3d(start: (float, float, float), originalFunc):
    x = start[0]
    y = start[1]
    z = start[2]
    steps = [start]

    for _ in range(MAX_ITERATIONS):
        gradX, gradY, gradZ = nd.Gradient(originalFunc)([x, y, z])

        gamma_values = np.logspace(np.log10(0.001), np.log10(10000), num=100)
        function_values = [originalFunc([x - gamma * gradX, y - gamma * gradY, z - gamma * gradZ]) for gamma in gamma_values]
        min_index = np.argmin(function_values)
        step_size = gamma_values[min_index]

        diffX = step_size * gradX
        diffY = step_size * gradY
        diffZ = step_size * gradZ
        if np.abs(diffX) + np.abs(diffY) + np.abs(diffZ) < TOLERANCE:
            break
        x -= diffX
        y -= diffY
        z -= diffZ
        steps.append((x, y, z)) 
  
    return steps, [x, y, z]

def graph_3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    graphData = np.array(data)
    ax.scatter(graphData[:, 0], graphData[:, 1], graphData[:, 2], c='b', marker='o')

    # Set labels for each axis
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Show the plot
    plt.show()

def gradient_descent(start: (float, float), gradientX, gradientY, learn_rate):
    x = start[0]
    y = start[1]
    steps = [start]

    for _ in range(MAX_ITERATIONS):
        diffX = learn_rate * gradientX(x, y)
        diffY = learn_rate * gradientY(x, y)
        if np.abs(diffX) + np.abs(diffY) < TOLERANCE:
            break
        x -= diffX
        y -= diffY
        steps.append((x, y)) 
  
    return steps, (x, y)

def fastest_descent(start: (float, float), gradientX, gradientY, func):
    x = start[0]
    y = start[1]
    steps = [start]

    for _ in range(MAX_ITERATIONS):
        gradX = gradientX(x, y)
        gradY = gradientY(x, y)

        gamma_values = np.logspace(np.log10(0.1), np.log10(1000), num=100)
        function_values = [func(x - gamma * gradX, y - gamma * gradY) for gamma in gamma_values]
        min_index = np.argmin(function_values)
        step_size = gamma_values[min_index]

        diffX = step_size * gradX
        diffY = step_size * gradY
        if np.abs(diffX) + np.abs(diffY) < TOLERANCE:
            break
        x -= diffX
        y -= diffY
        steps.append((x, y)) 
  
    return steps, (x, y)

def nelder_mead(start: (float, float), func):
    simplex = []
    simplex.append(SimplexPoint(start[0], start[1]))
    simplex.append(SimplexPoint(start[0] + 0.9659258263, start[1] + 0.2588190451))
    simplex.append(SimplexPoint(start[0] + 0.2588190451, start[1] + 0.9659258263))
    
    steps = []

    for _ in range(MAX_ITERATIONS):
        steps.append((simplex[0].x, simplex[0].y))
        simplex[0].value = func(simplex[0].x, simplex[0].y)
        simplex[1].value = func(simplex[1].x, simplex[1].y)
        simplex[2].value = func(simplex[2].x, simplex[2].y)
        simplex = sorted(simplex, key=lambda x: x.value)

        if (abs(simplex[1].x - simplex[0].x) + abs(simplex[1].y - simplex[0].y)) < TOLERANCE:
            break

        simplexC = get_center(simplex[0], simplex[1])
        simplexR = get_reflection(simplexC, simplex[2])
        simplexR.value = func(simplexR.x, simplexR.y)
        
        valMax = simplex[2].value
        valMid = simplex[1].value
        valMin = simplex[0].value
        valR = simplexR.value

        if valR < valMin:
            # if best, expand
            simplexE = deform_simplex(simplexC, simplexR, 2)
            simplex[2] = simplexR if func(simplexE.x, simplexE.y) > valR else simplexE
        elif valR < valMid:
            # do nothing
            simplex[2] = simplexR
        elif valR < valMax:
            # if value not worst but not good, try shrinking
            simplexE = deform_simplex(simplexC, simplexR, 0.5)
            simplex[2] = simplexR if func(simplexE.x, simplexE.y) > valR else simplexE
        else:
            # if value worst, shrink the other way
            simplexE = deform_simplex(simplexC, simplexR, -0.5)
            simplex[2] = simplexR if func(simplexE.x, simplexE.y) > valR else simplexE

    return steps, (simplex[0].x, simplex[0].y)

def show_graph(values):
    x, y = zip(*values)

    for i, (xi, yi) in enumerate(values):
        plt.text(xi, yi, str(i), ha='center', va='bottom', fontsize=8, color='black')

    plt.plot(x, y)
    plt.xlim(0, 1) 
    plt.ylim(-0.1, 1.1)

    plt.grid()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

class SimplexPoint:
    def __init__(self, x: float, y: float):
        self.value = 0
        self.x = x
        self.y = y

def get_center(first: SimplexPoint, second: SimplexPoint):
    return SimplexPoint((first.x + second.x) / 2, (first.y + second.y) / 2)

def get_reflection(center: SimplexPoint, other: SimplexPoint):
    return SimplexPoint(2 * center.x - other.x, 2 * center.y - other.y)

def deform_simplex(center: SimplexPoint, reflection: SimplexPoint, coeficient: float):
    return SimplexPoint(center.x + coeficient * (reflection.x - center.x), center.y + coeficient * (reflection.y - center.y))