import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(-1, 1, 50)
y = np.array([math.e ** (-x[i]) + x[i] - 2 if x[i] < 0 else -(math.e ** (-x[i]) + x[i]) for i in range(len(x))])
bound_points = np.array([
    [x[0], x[-1]],
    [y[0], y[-1]]
])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Move left y-axis and bottom x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.margins(0.2

plt.plot(x, y)
plt.scatter(*bound_points)
for xy in zip(*bound_points):
    ax.annotate('({:.3}, {:.3})'.format(*xy), xy=xy, textcoords='data')
plt.grid(True)
plt.show()
