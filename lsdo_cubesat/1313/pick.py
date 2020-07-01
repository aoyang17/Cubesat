import pickle
# import pyproj
# import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# path = "/home/lsdo/Cubesat/lsdo_cubesat/orbitdata/opt.01330.pkl"

path = "/home/lsdo/Cubesat/lsdo_cubesat/_data/opt.01465.pkl"

with open(path, 'rb') as f:
    info = pickle.load(f)
print(info['sunshade_cubesat_group.relative_orbit_state'].shape)

# B = np.arange(info['sunshade_cubesat_group.relative_orbit_state'].shape)
# print(B)
# plt.figure(figsize=(25, 5))
X = info['sunshade_cubesat_group.relative_orbit_state'][0, :]
Y = info['sunshade_cubesat_group.relative_orbit_state'][1, :]
plt.plot(X, Y)
# plt.ylim(ymin=0.1699)
# plt.ylim(ymax=0.170025)
# plt.xlim(xmin=0.0)
# plt.xlabel("time")
# plt.ylabel("reference_orbit_state")
# plt.legend()
# plt.grid()
# plt.savefig("Data_downloaded.png", dpi=120)
plt.show()

# print(info)
# X = {}
# X["reference_orbit_state"] = info["reference_orbit_state"]
# X["sunshade_cubesat_group.relative_orbit_state"] = info["sunshade_cubesat_group.relative_orbit_state"]

# np.savetxt('rundata/data.csv',
#            info["reference_orbit_state"],
#            header="reference_orbit_state",
#            delimiter=',')

# np.savetxt('rundata/data.csv',
#            info["sunshade_cubesat_group.relative_orbit_state"],
#            header='sunshade',
#            delimiter=',')

# with open('data.npy', 'wb') as f:
#     np.save(f, info["reference_orbit_state"])
#     np.save(f, info["sunshade_cubesat_group.relative_orbit_state"])
# with open('data.npy', 'wb') as f:
#     a = np.load(f)
# print(a)
# print(info_x['reference_orbit_state'])

# path = "/home/lsdo/Cubesat/lsdo_cubesat/_data/opt.00000.pkl"

# with open(path, 'rb') as f:
#     info_y = pickle.load(f)

# print(info_y['reference_orbit_state'])

# print(info_x['reference_orbit_state'] - info_y['reference_orbit_state'])

# print(info["sunshade_cubesat_group.relative_orbit_state"])
# X_refer = info["reference_orbit_state"][0, :]
# Y_refer = info["reference_orbit_state"][1, :]
# Z_refer = info["reference_orbit_state"][2, :]

# print(X_refer.shape[0])
print(info)
# # fig = plt.figure()
# # ax = Axes3D(fig)
# # # ax.plot(X, Y, Z)
# # # plt.plot(X_refer, Y_refer, Z_refer)
# # plt.plot(X_refer, Y_refer)
# # plt.show()

# # print(X_refer)
# # print(Y_refer)

# # ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
# # lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

# # lon, lat, alt = pyproj.transform(ecef,
# #                                  lla,
# #                                  X_refer,
# #                                  Y_refer,
# #                                  Z_refer,
# #                                  radians=True)
# # print(lon)
# # print(lat)
# # plt.figure(figsize=(15, 25))
# # ax = plt.axes(projection=ccrs.PlateCarree())
# # ax.stock_img()
# # plt.plot(lon, lat, 'k', transform=ccrs.Geodetic())
# # plt.show()

# X_sunshade = info["sunshade_cubesat_group.relative_orbit_state"][0, :]
# Y_sunshade = info["sunshade_cubesat_group.relative_orbit_state"][1, :]
# Z_sunshade = info["sunshade_cubesat_group.relative_orbit_state"][2, :]

# X_optics = info["optics_cubesat_group.relative_orbit_state"][0, :]
# Y_optics = info["optics_cubesat_group.relative_orbit_state"][1, :]
# Z_optics = info["optics_cubesat_group.relative_orbit_state"][2, :]

# X_detector = info["detector_cubesat_group.relative_orbit_state"][0, :]
# Y_detector = info["detector_cubesat_group.relative_orbit_state"][1, :]
# Z_detector = info["detector_cubesat_group.relative_orbit_state"][2, :]

# # X = X_refer + X_sunshade
# # Y = Y_refer + Y_sunshade
# # Z = Z_refer + Z_sunshade

# # distance_do = X_detector - X_optics
# # distance_os = X_optics - X_sunshade
# # print(np.shape(X_sunshade))
# mm = np.arange(X_sunshade.shape[0])
# print(mm.shape)
# plt.plot(mm, X_sunshade)
# plt.plot(mm, X_optics)
# plt.plot(mm, X_detector)
# plt.show()

# # X_optics = info["optics_cubesat_group.relative_orbit_state"][0, :]
# # Y_optics = info["optics_cubesat_group.relative_orbit_state"][1, :]

# # X_detector = info["detector_cubesat_group.relative_orbit_state"][0, :]
# # Y_detector = info["detector_cubesat_group.relative_orbit_state"][1, :]

# # # Y = info["sunshade_cubesat_group.relative_orbit_state"][4, :]
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot(X, Y, Z)
# plt.plot(X_refer / 1000000, Y_refer / 1000000)
# plt.plot(X, Y)
# plt.show()
