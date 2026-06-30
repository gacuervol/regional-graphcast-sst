import xarray as xr
import numpy as np
import re

def resize_lonxlat(dataset: xr.Dataset, new_lonxlat: tuple) -> xr.Dataset:
    """
    Resize the longitude and latitude dimensions of an xarray dataset.

    Parameters:
    dataset (xr.Dataset): The input xarray dataset.
    new_lonxlat (tuple): A tuple containing the new size of longitude and latitude dimensions, e.g., (lon_size, lat_size).

    Returns:
    xr.Dataset: The resized xarray dataset.

    Raises:
    ValueError: If the dataset does not contain longitude and latitude coordinates.

    Note:
    - The function interpolates the dataset to match the specified new size of the longitude and latitude dimensions.
    - When NaN values are present in the dataset, linear interpolation is used; otherwise, cubic interpolation is used.
    """
    # Find longitude and latitude coordinate names
    lon_name = [key for key in dataset.coords.keys() if re.match(r'^lon', key)][0]
    lat_name = [key for key in dataset.coords.keys() if re.match(r'^lat', key)][0]

    # Get start and end values of longitude and latitude
    lon_start, lon_end = dataset[lon_name].data[0], dataset[lon_name].data[-1]
    lat_start, lat_end = dataset[lat_name].data[0], dataset[lat_name].data[-1]

    # Extract new sizes of longitude and latitude
    lon_size, lat_size = new_lonxlat

    # Check if there are NaN values in the dataset
    there_nan = bool(dataset.isnull().any().to_array().data.any())

    # Interpolate the dataset based on whether NaN values are present
    if not there_nan:
        # Cubic interpolation when no NaN values are present
        return dataset.interp({lon_name: np.linspace(lon_start, lon_end, lon_size),
                               lat_name: np.linspace(lat_start, lat_end, lat_size)},
                               method="cubic")
    else:
        # Linear interpolation when NaN values are present
        return dataset.interp({lon_name: np.linspace(lon_start, lon_end, lon_size),
                               lat_name: np.linspace(lat_start, lat_end, lat_size)},
                               method="linear")