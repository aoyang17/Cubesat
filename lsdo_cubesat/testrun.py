import pickle
# import pyproj
# import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

path = '/home/lsdo/Cubesat/lsdo_cubesat/result_2/_data/opt.03811.pkl'
# path = '/home/lsdo/Cubesat/lsdo_cubesat/1314/_data_123/opt.03198.pkl'

with open(path, 'rb') as f:
    info = pickle.load(f)
# print(info)

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D((X_reference / 10), (Y_reference / 10), (Z_reference / 10))
ax.plot3D((X_reference + X_sunshade_relative),
          (Y_reference + Y_sunshade_relative),
          (Z_reference + Z_sunshade_relative))
# Axes3D.plot(X_reference, Y_reference, Z_reference)
plt.show()