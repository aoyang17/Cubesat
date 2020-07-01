import pickle
# import pyproj
# import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

path = "/home/lsdo/Cubesat/lsdo_cubesat/_data/opt.00219.pkl"

with open(path, 'rb') as f:
    info = pickle.load(f)
print(info['sunshade_cubesat_group.relative_orbit_state'].shape)

X_reference = info['reference_orbit_state'][0, :]
Y_reference = info['reference_orbit_state'][1, :]
Z_reference = info['reference_orbit_state'][2, :]

X_sunshade_relative = info['sunshade_cubesat_group.relative_orbit_state'][0, :]
Y_sunshade_relative = info['sunshade_cubesat_group.relative_orbit_state'][1, :]
Z_sunshade_relative = info['sunshade_cubesat_group.relative_orbit_state'][2, :]

X_detector_relative = info['detector_cubesat_group.relative_orbit_state'][0, :]
Y_detector_relative = info['detector_cubesat_group.relative_orbit_state'][1, :]
Z_detector_relative = info['detector_cubesat_group.relative_orbit_state'][2, :]

X_optics_relative = info['optics_cubesat_group.relative_orbit_state'][0, :]
Y_optics_relative = info['optics_cubesat_group.relative_orbit_state'][1, :]
Z_optics_relative = info['optics_cubesat_group.relative_orbit_state'][2, :]

time = np.arange((X_sunshade_relative.shape[0]))
# print(time)

X = X_reference / 1000.0 + X_sunshade_relative
Y = Y_reference / 1000.0 + Y_sunshade_relative

# fig = figure()
# ax = Axes3D(fig)
# ax.plot(X_reference / 100 + X_sunshade_relative,
#         Y_reference / 100 + Y_sunshade_relative,
#         Z_reference / 100 + Z_sunshade_relative)

# ax.plot(X_reference, Y_reference, Z_reference)

# import matplotlib.pyplot as plt
# import numpy as np
# import math

x = np.linspace(-10, 10, 100)


def sigmoid(x):
    return 1 / (1 + np.exp(-100 * x))


reference_matrix = info["reference_orbit_state"][:3, :]
r_orbit = np.linalg.norm(reference_matrix, ord=1, axis=0)

optics_matrix = info["optics_cubesat_group.relative_orbit_state"][:3, :]
r_optics = np.linalg.norm(optics_matrix, ord=1, axis=0)

detector_matrix = info["detector_cubesat_group.relative_orbit_state"][:3, :]
r_detector = np.linalg.norm(detector_matrix, ord=1, axis=0)

sunshade_matrix = info["sunshade_cubesat_group.relative_orbit_state"][:3, :]
r_sunshade = np.linalg.norm(sunshade_matrix, ord=1, axis=0)

time = np.arange(r_orbit.shape[0])

theta = 2 * np.pi / (r_orbit.shape[0]) * time

print(reference_matrix)

sns.set()

# plt.plot(time, r_orbit)
# plt.plot(time, r_optics)
# plt.plot(time[:5], (r_orbit + r_sunshade * 1e5)[:5], label='sunshade')
# plt.plot(time[:5], (r_orbit + r_optics * 1e5)[:5], label='optics')
# plt.plot(time[:5], (r_orbit + r_detector * 1e5)[:5], label='detector')
plt.plot(time, np.abs(r_sunshade - r_detector), label='sunshade-detector')
plt.plot(time, np.abs(r_optics - r_sunshade), label='optics-sunshade')
plt.plot(time, np.abs(r_optics - r_detector), label='optics-detector')
plt.plot(time, np.abs(r_detector - r_sunshade), label='detector-sunshade')
# plt.plot(time, r_detector - r_optics, label='detector-optics')
# plt.plot(time, r_sunshade - r_detector, label='sunshade-detector')
# plt.plot(time, np.abs(r_sunshade - r_optics), label='sunshade-optics')
# plt.plot(time, r_orbit / 10000 + r_detector)
plt.legend()
plt.show()
# ax = plt.subplot(121, projection='polar')
# ax.plot(theta, r_orbit / 10000)
# ax.plot(theta, r_optics + r_orbit / 10000)

# # plt.plot(time, r_orbit / 10000)
# # plt.plot(time, r_orbit / 10000)
# plt.show()

# X_detector_new = sigmoid(X_detector_relative)
# Y_detector_new = sigmoid(Y_detector_relative)
# Z_detector_new = sigmoid(Z_detector_relative)

# X_sunshade_new = sigmoid(X_sunshade_relative)
# Y_sunshade_new = sigmoid(Y_sunshade_relative)
# Z_sunshade_new = sigmoid(Z_sunshade_relative)

# plt.plot(x, z)
# plt.xlabel("x")
# plt.ylabel("Sigmoid(X)")
# plt.show()

# ax.plot(X_reference / 1000 + X_detector_relative,
#         Y_reference / 1000 + Y_detector_relative,
#         Z_reference / 1000 + Z_detector_relative)

# plt.plot(Y_reference / 100000 + Y_detector_relative * 100,
#          Z_reference / 100000 + Y_detector_relative * 100)

# plt.plot(X, Y)
# plt.ylim(ymin=0.1699)
# plt.ylim(ymax=0.170025)
# plt.xlim(xmin=0.0)
# plt.xlabel("time")
# plt.ylabel("reference_orbit_state")
# plt.legend()
# plt.grid()
# plt.savefig("Data_downloaded.png", dpi=120)
# plt.show()