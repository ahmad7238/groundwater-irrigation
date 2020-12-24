"""This code solves 2D Unconfined problem using Implicit scheme

Important Note: The code employs Cell Vertex scheme"""

import numpy as np
import matplotlib.pyplot as plt
import time
"Setting"
lx = 100
ly = 100
nx_max = 51
ny_max = 51
dx = lx/(nx_max-1)
dy = ly/(ny_max-1)

k = 0.001/60        # Saturated Hydraulic Conductivity
s_y = 0.002         # Specific Yield

elapsedtime = 0
totaltime = 86400.1
dt = 20

max_iter = 100
max_error = 0.0001

"Mesh Generation"
x= np.zeros([nx_max, ny_max])
for i in range(1, nx_max):
    for j in range(ny_max):
        x[i][j] = x[i][j-1] + dx

y = np.copy(x)
for j in range(1, ny_max):
    for i in range(nx_max):
        y[i][j] = y[i][j-1] + dy

print()