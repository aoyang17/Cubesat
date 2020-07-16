import matplotlib.pyplot as plt
plt.ion()

import cartopy.crs as ccrs

import pandas as pd

dates = pd.date_range(start="2017-12-11 00:00", periods=1000, freq="30S")

latlon = pd.DataFrame(index=dates, columns=["lat", "lon"])

for date in dates:
    lat, lon, _ = predictor.get_position(date).position_11h