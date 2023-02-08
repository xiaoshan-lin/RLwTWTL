import matplotlib.pyplot as plt
import numpy as np
import math
A=0.5
B=0.1
C=0.1
max_prob = 1.1
episodes = 100000
def epsilon(time):
    standardized_time=(time-A*episodes)/(B*episodes)
    cosh=np.cosh(math.exp(-standardized_time))
    epsilon=max_prob-(1/cosh+(time*C/episodes))
    return epsilon

y = []
for t in range(episodes):
    y.append(epsilon(t))

plt.plot(list(range(episodes)), y)
plt.show()
