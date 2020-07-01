import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pyproj
import pickle
import pandas as pd


def XYZ_2_LLA(X, Y, Z):

    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    lon, lat, alt = pyproj.transform(ecef, lla, X, Y, Z, radians=True)

    lon = 180 / np.pi * lon
    lat = 180 / np.pi * lat
    alt = 180 / np.pi * alt

    return lon, lat, alt

