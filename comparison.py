import numpy  as np
import matplotlib.pyplot as plt

plt.figure()
plt.title('ks=2 kt=4')

data = np.load('./out_numerical/eigenvalues_2_4_10000.npy')
vals, bins = np.histogram(data, bins=200, density=True)
plt.plot(bins[1:], vals, label='Numerical')

data = np.load('./out_popdyn/2_4_100000.npy')
x, y = data.T
plt.plot(x, y, label='Pop Dyn')

plt.legend()
plt.show()

