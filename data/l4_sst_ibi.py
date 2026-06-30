# Get the current user name
import os, sys
# Obtener el nombre del usuario actual
if sys.platform.startswith("linux"):
    usuario_actual = os.path.expanduser("~").split('/')[-1]
elif sys.platform.startswith("win"):
    usuario_actual = os.path.expanduser("~").split('\\')[-1]
else:
    raise EnvironmentError("Unsupported OS")

import numpy as np
import xarray as xr
import copernicusmarine
import matplotlib.pyplot as plt

sys.path.append(f"/home/{usuario_actual}/repo/src") 
from art1_tools import interpolation
from art1_tools import plot_utils

print(copernicusmarine.__version__)
username = 
password = 
copernicusmarine.login(username=username, password=password)
# Load Copernicus data
# https://data.marine.copernicus.eu/product/SST_ATL_SST_L4_REP_OBSERVATIONS_010_026/download?dataset=cmems-IFREMER-ATL-SST-L4-REP-OBS_FULL_TIME_SERIE_202012
satelite = "cmems-IFREMER-ATL-SST-L4-REP-OBS_FULL_TIME_SERIE"
min_lon, max_lon = (-20.97, -5.975) # IBI solo va hasta esta lon: -20.97
min_lat, max_lat = (19.55, 34.525)

# Load xarray dataset
SST_SAT = copernicusmarine.open_dataset(dataset_id=satelite,
                                        minimum_longitude=min_lon, 
                                        maximum_longitude=max_lon,
                                        minimum_latitude=min_lat,
                                        maximum_latitude=max_lat,
                                        )
graphcast_data_path = f'/home/{usuario_actual}/repo/data/'
SST_SAT.to_netcdf(graphcast_data_path + "IBI_SST_L4_FULL.nc")
