# Import libraries
import xarray as xr
import copernicusmarine
print(f"Copernicus marine toolbox: {copernicusmarine.__version__}")

# Get the current user name
import os, sys
# Obtener el nombre del usuario actual
if sys.platform.startswith("linux"):
    usuario_actual = os.path.expanduser("~").split('/')[-1]
elif sys.platform.startswith("win"):
    usuario_actual = os.path.expanduser("~").split('\\')[-1]
else:
    raise EnvironmentError("Unsupported OS")

# Download Copernicus data
# https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description
# dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
# https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description
dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m"
variables=["thetao"]
min_lon, max_lon = (-20.97, -5.975) # IBI solo va hasta esta lon: -20.97
min_lat, max_lat = (19.55, 34.525)
start_datetime = "2017-01-03"
end_datetime = "2021-01-09"
output_directory = f"/home/{usuario_actual}/repo/data/"
output_filename = "reanalysis.nc"
# Download data
copernicusmarine.subset(
    dataset_id=dataset_id,
    variables=variables,
    minimum_longitude=min_lon,
    maximum_longitude=max_lon,
    minimum_latitude=min_lat,
    maximum_latitude=max_lat,
    start_datetime=start_datetime,
    end_datetime=end_datetime,
    output_directory=output_directory,
    output_filename=output_filename, 
    )
# ds_file_name = 'cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m_thetao_20.92W-6.00W_19.58N-34.50N_0.49-5727.92m_2022-06-01-2024-12-08.nc'
glorys = xr.load_dataset(output_directory + output_filename)
# Get just the first depth level
glorys_depth_0 = glorys.isel(depth=0)
glorys_depth_0.to_netcdf(output_directory + 'glorys_depth_0.nc')
