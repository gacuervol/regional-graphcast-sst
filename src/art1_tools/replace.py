import numpy as np
import xarray as xr
from fuzzywuzzy import process  # Importing fuzzy string matching module

def replace_norm_matrices(norm_matrices: xr.Dataset, 
                          new_value: xr.DataArray,
                          vars_to_replace: xr.Dataset) -> xr.Dataset:
    """
    Replace values in an xarray Dataset based on approximate string matching of variable names.

    Parameters:
    - norm_matrices (xr.Dataset): An xarray Dataset containing normalized matrices.
    - new_value (xr.DataArray): An xarray DataArray representing the new value to replace in `norm_matrices`.
    - vars_to_replace (xr.Dataset): An xarray Dataset containing variables to be replaced in `norm_matrices`.

    Returns:
    - xr.Dataset: The modified `norm_matrices` Dataset after performing the replacements.
    """
    
    # Extract the norm value only once
    norm = float(new_value.data)

    # Loop through variables to be replaced
    for var_name, var in vars_to_replace.variables.items():
        var_dim = var.data.ndim  # Get the dimensionality of the variable
        variable_names = list(norm_matrices.data_vars)  # Get a list of variable names in the dataset
        
        # Find the closest variable name using fuzzy string matching
        closest_match = process.extractOne(var_name, variable_names)[0]
        
        n = norm_matrices[closest_match].data.size  # Get the size of the closest match variable

        # Replace values based on dimensionality
        if var_dim == 4:  # If variable has 4 dimensions
            norm_matrices[closest_match].data = norm  # Replace entire data with the new value
        elif var_dim == 5:  # If variable has 5 dimensions
            SST_norm_broadcast_n = [norm] * n  # Create a list of 'norm' values matching the 5th dimension size
            try:
                norm_matrices[closest_match].data = SST_norm_broadcast_n  # Broadcast the new value to match 5th dimension
            except ValueError:
                norm_matrices[closest_match].data = SST_norm_broadcast_n[0]
        else:
            pass  # Do nothing if the dimensionality doesn't match expected cases

    return norm_matrices  # Return the modified dataset

def replace_atmos_by_ocean(atmos_dataset: xr.Dataset,
                           ocean_var: np.array,
                           ) -> xr.Dataset:
    """
    Reemplaza variables atmosféricas en un dataset con datos de variables oceánicas.

    Esta función reemplaza las variables atmosféricas en un dataset con datos de variables oceánicas
    proporcionados como un arreglo numpy. Las variables en el dataset se reemplazan según su dimensionalidad.

    Args:
        atmos_dataset (xr.Dataset): Dataset de variables atmosféricas.
        ocean_var (np.array): Arreglo numpy de datos de variables oceánicas.

    Returns:
        xr.Dataset: Dataset actualizado con las variables reemplazadas por datos oceánicos.

    """
    for key in list(atmos_dataset.variables):
        if len(atmos_dataset[key].data.shape) == 5:
            # Repetir el arreglo oceánico a lo largo de la dimensión temporal del dataset
            ocean_var_5dims = np.repeat(ocean_var[:, :, np.newaxis, :, :], 13, axis=2)
            # Asignar los datos oceánicos al dataset atmosférico para variables con 5 dimensiones
            atmos_dataset[key].data = ocean_var_5dims
        elif len(atmos_dataset[key].data.shape) == 4:
            # Asignar los datos oceánicos al dataset atmosférico para variables con 4 dimensiones
            atmos_dataset[key].data = ocean_var
        else:
            pass  # Ignorar variables con otras dimensionalidades
    return atmos_dataset