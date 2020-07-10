import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyproj
import pickle
import pandas as pd
from PIL import Image

from lsdo_viz.api import BaseViz, Frame
from lsdo_viz.api import write_stl_triangles, write_tecplot_dat_curve, write_stl_structured_list
from lsdo_viz.api import get_sphere_triangulation, get_earth_triangulation, write_paraview

import seaborn as sns

sns.set(style='darkgrid')

earth_radius = 6371.
cubesat_names = ['sunshade', 'optics', 'detector']

time = 1501


def XYZ_2_LLA(X, Y, Z):

    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    lon, lat, alt = pyproj.transform(ecef, lla, X, Y, Z, radians=True)

    lon = 180 / np.pi * lon
    lat = 180 / np.pi * lat
    alt = 180 / np.pi * alt

    return lon, lat, alt


def datasort(lon, lat, alt):
    A = np.array([lon, lat, alt])
    B = A.T
    C = B[np.lexsort(B[:, ::-1].T)]
    return C


def viz(X, Y, Z):
    lon, lat, alt = XYZ_2_LLA(X, Y, Z)
    matrix = datasort(lon, lat, alt)
    return matrix


class Viz(BaseViz):
    def setup(self):
        self.frame_name_format = 'output_{}'

        self.add_frame(
            Frame(
                height_in=20.,
                width_in=25.,
                nrows=5,
                ncols=15,
                wspace=0.4,
                hspace=0.4,
            ), 1)

    def plot(self, data_dict_list, ind, video=False):
        import matplotlib.image as mpimg

        position_km_dict = dict()
        reference_orbit_dict = dict()
        relative_orbit_dict = dict()

        for cubesat_name in cubesat_names:
            reference_orbit = data_dict_list[ind]['reference_orbit_state']

            relative_orbit = data_dict_list[ind][
                '{}_cubesat_group.relative_orbit_state'.format(cubesat_name)]
            thrust_3xn = data_dict_list[ind][
                '{}_cubesat_group.thrust_3xn'.format(cubesat_name)]
            propellant = data_dict_list[ind][
                '{}_cubesat_group.propellant_mass'.format(cubesat_name)]
            data = data_dict_list[ind]['{}_cubesat_group.Data'.format(
                cubesat_name)]
            data_rate = data_dict_list[ind][
                '{}_cubesat_group.Download_rate'.format(cubesat_name)]
            roll = data_dict_list[ind]['{}_cubesat_group.roll'.format(
                cubesat_name)]
            pitch = data_dict_list[ind]['{}_cubesat_group.pitch'.format(
                cubesat_name)]
            P_comm = data_dict_list[ind]['{}_cubesat_group.P_comm'.format(
                cubesat_name)]
            propellant = data_dict_list[ind][
                '{}_cubesat_group.propellant_mass'.format(cubesat_name)]
            GSdist = data_dict_list[ind]['{}_cubesat_group.GSdist'.format(
                cubesat_name)]
            CommLOS = data_dict_list[ind]['{}_cubesat_group.CommLOS'.format(
                cubesat_name)]
            mass_flow_rate = data_dict_list[ind][
                '{}_cubesat_group.mass_flow_rate'.format(cubesat_name)]
            thrust_scalar = data_dict_list[ind][
                '{}_cubesat_group.thrust_scalar'.format(cubesat_name)]
            position = data_dict_list[ind]['{}_cubesat_group.position'.format(
                cubesat_name)]
            velocity = data_dict_list[ind]['{}_cubesat_group.velocity'.format(
                cubesat_name)]
            # antenna_angle = data_dict_list[ind][
            #     '{}_cubesat_group.antAngle'.format(cubesat_name)]

        normal_distance_sunshade_detector = data_dict_list[ind][
            'normal_distance_sunshade_detector_mm']
        normal_distance_optics_detector = data_dict_list[ind][
            'normal_distance_optics_detector_mm']
        distance_sunshade_optics = data_dict_list[ind][
            'distance_sunshade_optics_mm']
        distance_optics_detector = data_dict_list[ind][
            'distance_optics_detector_mm']
        masked_normal_distance_sunshade_detector = data_dict_list[ind][
            'masked_normal_distance_sunshade_detector_mm_sq_sum']
        masked_normal_distance_optics_detector = data_dict_list[ind][
            'masked_normal_distance_optics_detector_mm_sq_sum']
        masked_distance_sunshade_optics = data_dict_list[ind][
            'masked_distance_sunshade_optics_mm_sq_sum']
        masked_distance_optics_detector = data_dict_list[ind][
            'masked_distance_optics_detector_mm_sq_sum']
        sunshade_relative = data_dict_list[ind][
            'sunshade_cubesat_group.relative_orbit_state_sq_sum']
        optics_relative = data_dict_list[ind][
            'optics_cubesat_group.relative_orbit_state_sq_sum']
        detector_relative = data_dict_list[ind][
            'detector_cubesat_group.relative_orbit_state_sq_sum']
        obj = data_dict_list[ind]['obj']

        # optics_matrix = data_dict_list[ind][
        #     "optics_cubesat_group.relative_orbit_state"][:3, :]
        # r_optics = np.linalg.norm(optics_matrix, ord=1, axis=0)

        # detector_matrix = data_dict_list[ind][
        #     "detector_cubesat_group.relative_orbit_state"][:3, :]
        # r_detector = np.linalg.norm(detector_matrix, ord=1, axis=0)

        # sunshade_matrix = data_dict_list[ind][
        #     "sunshade_cubesat_group.relative_orbit_state"][:3, :]
        # r_sunshade = np.linalg.norm(sunshade_matrix, ord=1, axis=0)
        self.get_frame(1).clear_all_axes()

        with self.get_frame(1)[0, 0:3] as ax:

            data = data_dict_list[ind]['{}_cubesat_group.Data'.format(
                cubesat_name)]
            # print(data.shape)

            num_times = np.arange(time)
            # data_rate.reshape((1, 1501))
            sns.lineplot(x=num_times, y=data[0, :], ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'detector_cubesat_group.Data',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('Data_downloaded')

        with self.get_frame(1)[1, 0:3] as ax:

            data_rate = data_dict_list[ind][
                '{}_cubesat_group.Download_rate'.format(cubesat_name)]
            # print(data.shape)

            num_times = np.arange(time)
            # data_rate.reshape((1, 1501))
            sns.lineplot(x=num_times, y=data_rate, ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'detector_cubesat_group.Download_rate',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('Data_rate')

        with self.get_frame(1)[0, 12:15] as ax:

            propellant = data_dict_list[ind][
                '{}_cubesat_group.propellant_mass'.format(cubesat_name)]
            # print(data_rate.shape)

            num_times = np.arange(time)
            sns.lineplot(x=num_times, y=propellant[0, :], ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'detector_cubesat_group.propellant_mass',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('Propellant')

        with self.get_frame(1)[1, 12:15] as ax:

            mass_flow_rate = data_dict_list[ind][
                '{}_cubesat_group.mass_flow_rate'.format(cubesat_name)]
            # print(data_rate.shape)

            num_times = np.arange(time)
            sns.lineplot(x=num_times, y=mass_flow_rate, ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'detector_cubesat_group.mass_flow_rate',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('Mass_flow')

        with self.get_frame(1)[0, 4:11] as ax:
            ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
            lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

            reference_orbit = data_dict_list[ind]['reference_orbit_state']
            matrix_reference = viz(reference_orbit[0, :],
                                   reference_orbit[1, :],
                                   reference_orbit[2, :])

            path = "/home/lsdo/Cubesat/lsdo_cubesat/map/world.jpg"
            earth = mpimg.imread(path)
            # img = Image.open(path)
            ax.imshow(earth, extent=[-180, 180, -100, 100], aspect='auto')
            ax.plot(matrix_reference[:, 0],
                    matrix_reference[:, 1],
                    linewidth='1',
                    color='yellow')
            # sns.lineplot(x=matrix_reference[:, 0],
            #              y=matrix_reference[:, 1],
            #              linewidth='1',
            #              color='yellow',
            #              ax=ax)
            ax.scatter(-117.2340, 32.8801, marker="p", label="UCSD")
            ax.scatter(-88.2272, 40.1020, marker="p", label="UIUC")
            ax.scatter(-84.3963, 33.7756, marker="p", label="Georgia")
            ax.scatter(-109.533691, 46.9653, marker="p", label="Montana")

            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
            # ax.title("Trajectory of VISORS Satellite")

        with self.get_frame(1)[1, 4:11] as ax:
            ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
            lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

            reference_orbit = data_dict_list[ind]['reference_orbit_state']
            matrix_reference = viz(reference_orbit[0, :],
                                   reference_orbit[1, :],
                                   reference_orbit[2, :])

            sns.lineplot(x=matrix_reference[:, 0],
                         y=matrix_reference[:, 2],
                         ax=ax)

            ax.set_xlabel("longitude")
            ax.set_ylabel("altitude")

        with self.get_frame(1)[2, 0:3] as ax:
            num_times = np.arange(time)

            normal_distance_sunshade_detector = data_dict_list[ind][
                'normal_distance_sunshade_detector_mm']

            # print(normal_distance_sunshade_detector.shape)
            normal_distance_sunshade_detector.reshape((1, time))
            sns.lineplot(x=num_times,
                         y=normal_distance_sunshade_detector,
                         ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'normal_distance_sunshade_detector_mm',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.axvline(1000, color='red')
            ax.axvline(1400, color='red')
            ax.set_xlabel('num_times')
            ax.set_ylabel('alignment_s_d')

        with self.get_frame(1)[2, 4:7] as ax:
            num_times = np.arange(time)

            normal_distance_optics_detector = data_dict_list[ind][
                'normal_distance_optics_detector_mm']

            normal_distance_optics_detector.reshape((1, time))
            sns.lineplot(x=num_times, y=normal_distance_optics_detector, ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'normal_distance_optics_detector_mm',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.axvline(1000, color='red')
            ax.axvline(1400, color='red')
            ax.set_xlabel('num_times')
            ax.set_ylabel('alignment_o_d')

        with self.get_frame(1)[2, 8:11] as ax:
            num_times = np.arange(time)

            distance_sunshade_optics = data_dict_list[ind][
                'distance_sunshade_optics_mm']

            distance_sunshade_optics.reshape((1, time))
            sns.lineplot(x=num_times, y=distance_sunshade_optics, ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'distance_sunshade_optics_mm',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('seperation_s_o')

        with self.get_frame(1)[2, 12:15] as ax:
            num_times = np.arange(time)

            distance_optics_detector = data_dict_list[ind][
                'distance_optics_detector_mm']

            distance_optics_detector.reshape((1, time))
            sns.lineplot(x=num_times, y=distance_optics_detector, ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'distance_optics_detector_mm',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('seperation_o_d')

        with self.get_frame(1)[3, 0:3] as ax:
            num_times = np.arange(time)
            roll = data_dict_list[ind]['detector_cubesat_group.roll']
            # print(roll_rate.shape)
            roll.reshape((1, time))
            sns.lineplot(x=num_times, y=roll, ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'detector_cubesat_group.roll',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('Roll_scalar')
            # ax.get_xaxis().set_ticks([])

        with self.get_frame(1)[3, 4:7] as ax:
            num_times = np.arange(time)
            pitch = data_dict_list[ind]['detector_cubesat_group.pitch']
            print(pitch.shape)
            pitch.reshape((1, time))
            sns.lineplot(x=num_times, y=pitch, ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'detector_cubesat_group.pitch',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('Pitch_scalar')
            # ax.get_xaxis().set_ticks([])

        with self.get_frame(1)[3, 8:11] as ax:

            thrust_scalar = data_dict_list[ind][
                '{}_cubesat_group.thrust_scalar'.format(cubesat_name)]
            # print(data_rate.shape)

            num_times = np.arange(time)
            thrust_scalar.reshape((1, time))
            sns.lineplot(x=num_times, y=thrust_scalar, ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'detector_cubesat_group.thrust_scalar',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('Thrust_scalar')

        with self.get_frame(1)[3, 12:15] as ax:

            velocity = data_dict_list[ind]['{}_cubesat_group.velocity'.format(
                cubesat_name)]
            # print(data_rate.shape)
            velocity = np.linalg.norm(velocity, ord=1, axis=0)
            num_times = np.arange(time)
            velocity.reshape((1, time))
            sns.lineplot(x=num_times, y=velocity, ax=ax)
            # if video:
            #     ax.set_ylim(
            #         self.get_limits(
            #             'detector_cubesat_group.velocity',
            #             lower_margin=0.1,
            #             upper_margin=0.1,
            #         ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('Velocity')

        with self.get_frame(1)[4, 0:3] as ax:

            P_comm = data_dict_list[ind]['{}_cubesat_group.P_comm'.format(
                cubesat_name)]
            # print(data_rate.shape)

            num_times = np.arange(time)
            P_comm.reshape((1, time))
            sns.lineplot(x=num_times, y=P_comm, ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'detector_cubesat_group.P_comm',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('P_comm')

        with self.get_frame(1)[4, 4:7] as ax:

            GSdist = data_dict_list[ind]['{}_cubesat_group.GSdist'.format(
                cubesat_name)]
            # print(data_rate.shape)

            num_times = np.arange(time)
            GSdist.reshape((1, time))
            sns.lineplot(x=num_times, y=GSdist, ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'detector_cubesat_group.GSdist',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('GSdist')

        with self.get_frame(1)[4, 8:11] as ax:

            CommLOS = data_dict_list[ind]['{}_cubesat_group.CommLOS'.format(
                cubesat_name)]
            # print(data_rate.shape)

            num_times = np.arange(time)
            CommLOS.reshape((1, time))
            sns.lineplot(x=num_times, y=CommLOS, ax=ax)
            if video:
                ax.set_ylim(
                    self.get_limits(
                        'detector_cubesat_group.CommLOS',
                        lower_margin=0.1,
                        upper_margin=0.1,
                    ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('CommLOS')

        with self.get_frame(1)[4, 12:15] as ax:

            position = data_dict_list[ind]['{}_cubesat_group.position'.format(
                cubesat_name)]
            # print(data_rate.shape)
            position = np.linalg.norm(position, ord=1, axis=0)
            num_times = np.arange(time)
            position.reshape((1, time))
            sns.lineplot(x=num_times, y=position, ax=ax)
            # if video:
            #     ax.set_ylim(
            #         self.get_limits(
            #             'detector_cubesat_group.position',
            #             lower_margin=0.1,
            #             upper_margin=0.1,
            #         ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('Position')

        self.get_frame(1).write()