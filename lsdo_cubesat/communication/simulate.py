import numpy as np
from openmdao.api import Problem, IndepVarComp, Group, ExecComp

from lsdo_cubesat.attitude.rot_mtx_b_i_comp import RotMtxBIComp
from lsdo_cubesat.communication.Antenna_rot_mtx import AntennaRotationMtx
from lsdo_cubesat.communication.Antenna_rotation import AntRotationComp
from lsdo_cubesat.communication.Comm_Bitrate import BitRateComp
from lsdo_cubesat.communication.Comm_distance import StationSatelliteDistanceComp
from lsdo_cubesat.communication.Comm_LOS import CommLOSComp
from lsdo_cubesat.communication.Comm_VectorBody import VectorBodyComp
from lsdo_cubesat.communication.GSposition_ECEF_comp import GS_ECEF_Comp
from lsdo_cubesat.communication.GSposition_ECI_comp import GS_ECI_Comp
# from lsdo_cubesat.communication.rot_mtx_ECI_EF_comp import RotMtxECIEFComp
from lsdo_cubesat.communication.Vec_satellite_GS_ECI import Comm_VectorECI
from lsdo_cubesat.communication.Comm_vector_antenna import AntennaBodyComp
from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.communication.Earth_spin_comp import EarthSpinComp
from lsdo_cubesat.communication.Earthspin_rot_mtx import EarthspinRotationMtx
# from lsdo_cubesat.communication.Ground_comm import Groundcomm

num_times = 1501
num_cp = 300
step_size = 95 * 60 / (num_times - 1)
B = np.arange(0, num_times, 1)
Re = 6378.137

times = np.linspace(0., step_size * (num_times - 1), num_times)

# r_e2g_E = np.loadtxt('/home/lsdo/Cubesat/lsdo_cubesat/rundata/r_e2g_E.csv')
C = np.loadtxt('/home/lsdo/Cubesat/lsdo_cubesat/rundata/orbitstate.csv')
group = Group()

comp = IndepVarComp()
comp.add_output('times', val=np.arange(num_times))
comp.add_output('lon', val=-83.7264, units='rad')
comp.add_output('lat', val=42.2708, units='rad')
comp.add_output('alt', val=0.256, units='km')
comp.add_output('orbit_state_km', C, units=None)

group.add_subsystem('Inputcomp', comp, promotes=['*'])

group.add_subsystem('EarthSpinComp',
                    EarthSpinComp(num_times=num_times),
                    promotes=['*'])

group.add_subsystem('GS_ECEF_Comp',
                    GS_ECEF_Comp(num_times=num_times),
                    promotes=['*'])

group.add_subsystem('Rot_ECI_EF',
                    EarthspinRotationMtx(num_times=num_times),
                    promotes=['*'])

group.add_subsystem('r_e2g_I',
                    GS_ECI_Comp(num_times=num_times),
                    promotes=['*'])

group.add_subsystem('r_b2g_I',
                    Comm_VectorECI(num_times=num_times),
                    promotes=['*'])

prob = Problem()
prob.model = group
prob.setup(check=True)
prob.run_model()
prob.model.list_outputs()
print(prob['r_e2g_E'])
# print(prob['r_e2g_I'])
prob.check_partials(compact_print=True)

r_e2g_E = prob['r_e2g_E']
r_e2g_I = prob['r_e2g_I']
r_b2g_I = prob['r_b2g_I']

R = np.linalg.norm(r_e2g_E, ord=2, axis=0)
print(R)
R_1 = np.linalg.norm(C[:3, :], ord=2, axis=0)
print(R_1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1 + 2 * np.exp(-x)) / (np.exp(-x) + 1)**2


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(r_b2g_I[0, :], r_b2g_I[1, :], r_b2g_I[2, :], 'blue')
ax.plot3D(r_e2g_I[0, :], r_e2g_I[1, :], r_e2g_I[2, :], 'gray')
ax.plot3D(C[0, :], C[1, :], C[2, :], 'black')
ax.scatter(0, 0, 0, '*')
plt.show()

proj_A = np.sum(r_b2g_I * r_e2g_I, axis=0) / Re
CommLOS = np.ones(num_times)

CommLOS = sigmoid(proj_A - 1e5 - 1e4)

# proj_B = np.sum(C[:3, :] * r_e2g_I, axis=0) / Re

# projlist = []
# for i in range(0, num_times):

#     proj = np.dot(r_b2g_I[:, i], r_e2g_I[:, i]) / Re
#     projlist.append(proj)
plt.figure(figsize=(25, 5))

plt.plot(B, CommLOS, label='A')
# plt.plot(B, proj_B, label="B")
# plt.plot(B, projlist, label='C')
# plt.plot(B, r_e2g_I[2, :], label="Z")

plt.xlim(xmin=0.0)
plt.legend()
plt.grid()
plt.show()