import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns

Re = 6378.137

A = np.loadtxt("projection.csv", comments="#", delimiter=",", skiprows=1)
C = np.loadtxt('new_orbitstate.csv')
# print(A)
B = np.arange(A.shape[0])
plt.figure(figsize=(25, 5))
sns.set()
datarate = np.loadtxt("detector_data_rate.csv",
                      comments="#",
                      delimiter=",",
                      skiprows=1)
data = np.loadtxt("detector_propellant.csv",
                  comments="#",
                  delimiter=",",
                  skiprows=1)

plt.plot(B, data)
plt.title("Propellant used in one cycle")
plt.show()

# r_b2g_I = np.loadtxt("optics_r_b2g_I.csv")
# r_e2g_I = np.loadtxt("optics_r_e2g_I.csv")

# r_e2g_E = np.loadtxt("r_e2g_E.csv")
# print(r_e2g_I)

# D = C[:3, :]

# CommLOS = np.zeros(1501)
# # for i in range(0, 1501):
# #     proj = np.dot(D[:, i], r_e2g_I[:, i]) / Re
# #     if proj < 0:
# #         CommLOS[i] = 0.
# #     elif proj > 20000:
# #         CommLOS[i] = 1.
# # else:# #     x = (proj - 0) / (-Rb - 0)

# def sigmoid_grad(x):
#     return (1 + 2 * np.exp(-x)) / (np.exp(-x) + 1)**2

# proj = np.sum(r_b2g_I * r_e2g_I, axis=0) / Re
# print(np.min(proj))
# proj_D = np.sum(D * r_e2g_I, axis=0) / Re

# CommLOS = sigmoid(proj - 1e5)

# # # C = np.loadtxt("orbitstate.csv")

# Optimality = np.array([
#     1.1e-3, 1.1e-3, 3.5e-4, 1.3e-4, 1.3e-4, 1.3e-4, 1.1e-4, 1.1e-4, 1.1e-4,
#     1.1e-4, 1.1e-4, 1.1e-4, 1.1e-4, 1.1e-4, 1.1e-4, 1.1e-4, 6.1e-5, 6.1e-5,
#     6.1e-5, 6.1e-5, 6.1e-5, 6.1e-5, 8.2e-5, 8.2e-5, 3.8e-5, 1.5e-5, 1.5e-5,
#     1.3e-5, 1.1e-5, 9.5e-6
# ])

# Step = np.arange(Optimality.shape[0])

# sns.set()
# plt.plot(Step, Optimality, label='Optimality')
# plt.hlines(1e-5, 0, 30, colors='r', linestyles='dashed')
# plt.title("Optimality with optimization steps")
# plt.legend()
# plt.show()

# # # print(C.shape)
# # D = 6378.137
# # D = np.linalg.norm(r_e2g_I, ord=1, axis=0)
# # E = np.linalg.norm(r_e2g_E, ord=1, axis=0)
# # print(D)
# # print(E)
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# ax.plot3D(r_b2g_I[0, :], r_b2g_I[1, :], r_b2g_I[2, :])
# ax.plot3D(r_e2g_I[0, :], r_e2g_I[1, :], r_e2g_I[2, :], 'gray')
# ax.plot3D(C[0, :], C[1, :], C[2, :], 'black')
# ax.scatter(0, 0, 0, '*')
# plt.show()

# plt.figure(figsize=(25, 5))

# # plt.plot(B, r_e2g_E[0, :], label='X')
# # plt.plot(B, r_e2g_E[1, :], label="Y")
# # plt.plot(B, r_e2g_E[2, :], label="Z")
# plt.plot(B, proj, label="1")
# plt.plot(B, proj_D, label="2")
# # plt.ylim(ymin=0.1699)
# # plt.ylim(ymax=0.170025)
# plt.xlim(xmin=0.0)
# plt.xlabel("time")
# plt.ylabel("Comm_LOS")
# plt.legend()
# plt.grid()
# plt.show()

# # plt.savefig("Data_downloaded.png", dpi=120)
# plt.show()

# with open('optics_propellant.csv', 'wb') as f:
#     a = np.load(f)

# print(a)

# np.savetxt("rundata/optics_propellant.csv",
#            prob['optics_cubesat_group.propellant_mass'],
#            header="optics_cubesat_group.propellant_mass",
#            delimiter=',')

# np.savetxt("rundata/detector_propellant.csv",
#            prob['detector_cubesat_group.propellant_mass'],
#            header="detector_cubesat_group.propellant_mass",
#            delimiter=',')

# np.savetxt("rundata/sunshade_propellant.csv",
#            prob['sunshade_cubesat_group.propellant_mass'],
#            header="sunshade_cubesat_group.propellant_mass",
#            delimiter=',')

# np.savetxt("rundata/optics_data.csv",
#            prob['optics_cubesat_group.Data'],
#            header="optics_cubesat_group.Data",
#            delimiter=',')

# np.savetxt("rundata/detector_data.csv",
#            prob['detector_cubesat_group.Data'],
#            header="detector_cubesat_group.Data",
#            delimiter=',')

# np.savetxt("rundata/sunshade_data.csv",
#            prob['sunshade_cubesat_group.Data'],
#            header="sunshade_cubesat_group.Data",
#            delimiter=',')