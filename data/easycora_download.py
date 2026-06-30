# First-party imports
import os
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Optional
from pprint import pprint

# Second-party imports
# Third-party imports
from pathlib import Path
import xarray as xr
from netCDF4 import Dataset
import copernicusmarine
from tqdm.notebook import trange, tqdm

# Load Copernicus data
credentials_file = "~/.copernicusmarine/.copernicusmarine-credentials"
# https://data.marine.copernicus.eu/product/INSITU_GLO_PHY_TS_DISCRETE_MY_013_001/description
dataset_id = "cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr"
# Load xarray dataset
copernicusmarine.get(dataset_id=dataset_id,
                    regex=".*201[7-9].*|.*2020.*", 
                    credentials_file=credentials_file,
                    )   

# module.py
# Global variables
# Define zone
MINIMUM_LATITUDE, MAXIMUM_LATITUDE = (19.55, 34.525)
MINIMUM_LONGITUDE, MAXIMUM_LONGITUDE = (-20.97, -5.975)

class DataType(Enum):
    """Tipos de geometría de datos según el PUM"""
    PR = "PR"  # Vertical Profiles
    TS = "TS"  # Time Series

class InstrumentCode(Enum):
    """Códigos de instrumentos oficiales del manual CMEMS-INS-PUM-013-001"""
    BT = "BT"  # Bottles
    CT = "CT"  # CTD
    DR = "DR"  # Drifter buoys
    FB = "FB"  # Ferry boxes
    GL = "GL"  # Gliders
    ML = "ML"  # Mini loggers
    MO = "MO"  # Moorings
    PF = "PF"  # Profilers
    SD = "SD"  # Sail drones
    SF = "SF"  # Scanfish
    SM = "SM"  # Sea mammals
    TS = "TS"  # Thermosalinograph
    TX = "TX"  # Thermistor chain
    XB = "XB"  # XBT, XCTD or MBT profiles
    XX = "XX"  # Not yet identified

@dataclass
class InSituDataPaths:
    """Class to handle in situ data"""
    path: Optional[str | Path]
    data_type: DataType
    instrument: InstrumentCode
    # Usamos field(init=False) para que no pida estos datos al crear la instancia
    domain_folders: List[str] = field(init=False)
    year_folders: List[str] = field(init=False)
    # Definimos el diccionario anidado correctamente
    folder_struct: Dict[str, Dict[str, List[Path]]] = field(init=False, default_factory=lambda: defaultdict(dict))

    def __post_init__(self):
        # 1. Asegurar que path sea objeto Path
        self.path = Path(self.path) if isinstance(self.path, str) else self.path
        
        # 2. Listar dominios (ej. 'mediterrane', 'arctic', 'global')
        self.domain_folders = list(path.name for path in self.path.iterdir())
        
        # 3. Listar años usando el primer dominio encontrado
        first_domain = self.path / self.domain_folders[0]
        self.year_folders = sorted([path.name for path in first_domain.iterdir() if path.is_dir()])
        
        # 4. Mapear la estructura
        self.map_struct()
    
    def map_struct(self):
        # Llenamos el defaultdict de diccionarios
        for domain in self.domain_folders:
            for year in self.year_folders:
                year_path = self.path / domain / year
                if year_path.exists():
                    # Guardamos la lista de archivos NetCDF (.nc)
                    self.folder_struct[domain][year] = sorted(
                        list(year_path.glob(f"*{self.data_type.value}_{self.instrument.value}.nc")),
                        key=lambda path: path.name.split('_')[2]
                        )

def is_file_in_zone(file_path: Path, minimum_longitude: float, maximum_longitude: float, minimum_latitude: float, maximum_latitude: float) -> List[bool]:
    """     
    Check if a NetCDF file contains data within a specified latitude and longitude range.
    This function is faster than using xarray for this purpose.
    
    Args:
        file_path (Path): Path to the NetCDF file.
        
    Returns:
        List[bool]: List of booleans indicating if each latitude and longitude point is within the specified range.
    """
    # Open the nc with netCDF4 (much faster than xarray for this)
    with Dataset(file_path, mode='r') as nc:
        # Assuming the variables are named 'LATITUDE' and 'LONGITUDE'
        # (Sometimes it's 'lat', 'lon', check your files with ncdump if it fails)
        
        # NOTE: If your files have global attributes like 'geospatial_lat_min', etc.
        # it's EVEN faster to read those attributes instead of the variables.
        # Here we read the variables for safety:
        lats = nc.variables['LATITUDE'][:]
        lons = nc.variables['LONGITUDE'][:]
        
        # Define the zone
        mask = (
            (lats >= MINIMUM_LATITUDE) & (lats <= MAXIMUM_LATITUDE) 
            & (lons >= MINIMUM_LONGITUDE) & (lons <= MAXIMUM_LONGITUDE)
            )
        
        return True if np.any(mask) else False

def get_valid_files(files: List[Path]) -> List[Path]:
    """     
    Avoid files that are not in the zone.
    
    Args:
        files (List[Path]): List of files to avoid.
        
    Returns:
        List[Path]: List of files to avoid.
    """
    # Filter files to avoid to load to memory files that are not in the zone
    presence_flag_fn = lambda file: is_file_in_zone(file, MINIMUM_LONGITUDE, MAXIMUM_LONGITUDE, MINIMUM_LATITUDE, MAXIMUM_LATITUDE)
    
    return list(filter(presence_flag_fn, files))

def remove_redundant_vars(ds: xr.Dataset) -> xr.Dataset:
    """     
    Remove structural metadata variables that cause redundancy 
    due to audit dimensions (N_PARAM, N_EV_PROF)
    
    Args:
        ds (xr.Dataset): Dataset to remove variables from.
        
    Returns:
        xr.Dataset: Dataset with variables removed.
    """
    vars_to_remove = [
        'STATION_PARAMETERS', 
        'REMOVED_PROFILES_DC_REFERENCE', 
        'ESTIMATED_STATION_PARAMETERS'
    ]
    return ds.drop_vars(vars_to_remove, errors='ignore')

def coord_subset(ds: xr.Dataset) -> pd.DataFrame:
    df = ds.to_dataframe().reset_index()
    df = df[
        df['LONGITUDE'].between(MINIMUM_LONGITUDE, MAXIMUM_LONGITUDE) &
        df['LATITUDE'].between(MINIMUM_LATITUDE, MAXIMUM_LATITUDE)
        ]   
    # Control NaN values
    assert df['TEMP'].isnull().sum() == 0, "There are NaN values in the temperature column"

    return df

def cols_subset(df: pd.DataFrame) -> pd.DataFrame:
    TEMP_cols_to_keep = [
        'TIME', 'LATITUDE', 'LONGITUDE',           # Where and When
        'DEPH', 'PRES',                            # Depth (are complementary)
        'TEMP',                                    # Your variable
        'TEMP_QC',                                 # Quality of your variable
        'PLATFORM_NUMBER',                         # Buoy identifier
        'INSTITUTION',
        'DEPH_QC', 'PRES_QC', 'TIME_QC', 'POSITION_QC' # Quality of coordinates (Optional but recommended)
        ]
    
    return df[TEMP_cols_to_keep].copy()

PATH = "/home/user/repo/data/INSITU_GLO_PHY_TS_DISCRETE_MY_013_001/cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr_202511/"
domain_folder = "global"
list_dfs = [] # List to store DataFrames

for year in trange(2017, 2021):
    data_paths = InSituDataPaths(
        PATH, 
        data_type=DataType.TS, 
        instrument=InstrumentCode.DR
        ).folder_struct[domain_folder][str(year)]

    for file in get_valid_files(data_paths):
        ds_insitu = xr.open_dataset(file)

        list_dfs.append(
            ds_insitu
            .sel(N_LEVELS=0) # Select the first level
            .pipe(remove_redundant_vars)
            .pipe(coord_subset)
            .pipe(cols_subset)
            )

# Concatenate all DataFrames
df_final = pd.concat(list_dfs, ignore_index=True)
df_final.to_csv("/home/user/repo/data/in_situ_drifters_canary_2017_2020.csv", index=False)
