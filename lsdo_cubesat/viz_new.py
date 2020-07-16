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

# using seaborn

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


def spread_out_orbit(rel_rel_scale, ref_rel_scale, ref, rel, others):
    """
    This function exaggerates the distance between a spacecraft's orbit
    and the reference orbit for the swarm. `ref` and `rel` are n-vectors
    and represent X, Y, or Z component of the position.
    """
    m_to_km = 1e3

    # spread out orbits relative to each other
    drel = rel - rel
    for other in others:
        drel += rel - other
    ex = rel - rel_rel_scale * drel

    # spread out orbits relative to reference orbit
    spread_ref_scale = ref_rel_scale * m_to_km
    exaggerated = ref + spread_ref_scale * ex
    return exaggerated


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
            ),
            1,
        )

    def plot(self, data_dict_list, ind, video=False):
        import matplotlib.image as mpimg

        reference_orbit = data_dict_list[ind]['reference_orbit_state']

        # FIXME: prints True; should print False
        print(reference_orbit is None)

        relative_orbit = dict()
        for cubesat_name in cubesat_names:
            print(cubesat_name)
            relative_orbit[cubesat_name] = data_dict_list[ind][
                '{}_cubesat_group.relative_orbit_state'.format(cubesat_name)]

            # FIXME: prints True; should print False
            print(data_dict_list[ind]['{}_cubesat_group.relative_orbit_state'.
                                      format(cubesat_name)] is None)

        rel_rel_scale = 500
        ref_rel_scale = 1

        x_tuple = [dict()] * len(cubesat_names)
        y_tuple = [dict()] * len(cubesat_names)
        z_tuple = [dict()] * len(cubesat_names)
        i = 0
        for cubesat_name in cubesat_names:
            # FIXME: error here because RHS is None type -- data is not loaded
            x_tuple[i] = relative_orbit[cubesat_name][0, :]
            y_tuple[i] = relative_orbit[cubesat_name][1, :]
            z_tuple[i] = relative_orbit[cubesat_name][2, :]
            i += 1

        position = dict()
        for cubesat_name in cubesat_names:
            position[cubesat_name][0, :] = spread_out_orbit(
                rel_rel_scale,
                ref_rel_scale,
                reference_orbit[0, :],
                relative_orbit[cubesat_name][0, :],
                x_tuple,
            )
            position[cubesat_name][0, :] = spread_out_orbit(
                rel_rel_scale,
                ref_rel_scale,
                reference_orbit[1, :],
                relative_orbit[cubesat_name][1, :],
                y_tuple,
            )
            position[cubesat_name][0, :] = spread_out_orbit(
                rel_rel_scale,
                ref_rel_scale,
                reference_orbit[2, :],
                relative_orbit[cubesat_name][2, :],
                z_tuple,
            )

        self.get_frame(1).clear_all_axes()

        with self.get_frame(1)[0, 0] as ax:

            for cubesat_name in cubesat_names:
                ax.plot(
                    position[cubesat_name][0, :],
                    position[cubesat_name][1, :],
                )

        with self.get_frame(1)[0, 1] as ax:

            for cubesat_name in cubesat_names:
                ax.plot(
                    position[cubesat_name][2, :],
                    position[cubesat_name][1, :],
                )

        with self.get_frame(1)[1, 0] as ax:

            for cubesat_name in cubesat_names:
                ax.plot(
                    position[cubesat_name][0, :],
                    position[cubesat_name][2, :],
                )
        self.get_frame(1).write()
