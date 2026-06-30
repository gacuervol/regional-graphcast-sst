import xarray as xr
import numpy as np
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class CompatibleData:
    """Class for compatible data"""
    name: str
    forecast_and_truth: Tuple[xr.Dataset, xr.Dataset]

    # Obtiene la fecha inicial de las predicciones
    def initial_date(self) -> np.datetime64:
        return self.forecast_and_truth[1].datetime.values[:, 0] - np.timedelta64(1, 'D')


class PredictionCompatibilityChecker:
    """Main class for checking compatibility between predictions and baseline.

    Usage:
    >>> baseline = xr.open_dataset("baseline.nc")
    >>> models = {
    ...     "model1": {
    ...         "name": "model1",
    ...         "predictions": xr.open_dataset("model1.nc")
    ...     },
    ...     "model2": {
    ...         "name": "model2",
    ...         "predictions": xr.open_dataset("model2.nc")
    ...     }
    ... }
    >>> checker = PredictionCompatibilityChecker(baseline, models)
    >>> compatible_data = checker.check_compatibility()


    Attributes:
    baseline: xarray.Dataset that acts as reference.
    models: Dictionary with the form {name_model: {"name": str, "predictions": xr.Dataset}}.
    compatible_data_store: List with the compatible data.
    """
    def __init__(self, baseline: xr.Dataset, models: Dict[str, Dict]):
        """
        Inicializa la clase con el baseline y un diccionario de modelos.
        
        :param baseline: xarray.Dataset que actúa como referencia.
        :param models: Diccionario con la forma {name_model: {"name": str, "predictions": xr.Dataset}}.
        """
        self.baseline = baseline
        self.models = models

    def check_compatibility(self):
        """
        Verifica la compatibilidad entre las predicciones de cada modelo y el baseline.
        Si son compatibles, almacena las tuplas (pred, base) en self.compatible_data.
        """
        compatible_data_store = [] # Almacena los datos compatibles
        for model in self.models:
            print(f":: Checking compatibility for: {model['name']}...")
            predictions = model['predictions']
            
            # Verifica que los tipos de datos coincidan
            if not self._types_match(predictions):
                raise ValueError(f"Los tipos de datos no coinciden para el modelo {model['name']}")
            # Verifica que las dimensiones coincidan
            if not self._dims_size_match(predictions):
                try:
                    predictions = PredictionCompatibilityChecker.split_in_batch(predictions, self.baseline.time.size + 2)
                except Exception as exc:
                    raise ValueError(
                        f"Numero de dimensiones no coinciden para el modelo {model['name']}: {list(self.baseline.dims)} vs {list(predictions.dims)}"
                        ) from exc
            # Verifica que los shapes coincidan
            if not self._shapes_match(predictions):
                try: 
                    if not self._lat_lon_match(predictions):
                        try:
                            new_lonxlat = (
                            self.baseline.dims[self.lon_name_baseline],
                            self.baseline.dims[self.lat_name_baseline],
                            )
                            predictions = PredictionCompatibilityChecker.downscaling(
                                predictions, 
                                new_lonxlat,
                                )
                            predictions = predictions.transpose(*tuple(self.baseline.dims)[:2], ...)
                        except ValueError:
                            raise ValueError(
                                f"""Las latitudes y longitudes no coinciden para baseline vs modelo {model['name']}: 
                                {self.lat_name_baseline}: {self.baseline.dims[self.lat_name_baseline]} vs {self.lat_name_pred}: {predictions.dims[self.lat_name_pred]}
                                {self.lon_name_baseline}: {self.baseline.dims[self.lon_name_baseline]} vs {self.lon_name_pred}: {predictions.dims[self.lon_name_pred]}
                                """)
                except Exception as exc:
                    raise ValueError(
                        f"Los shapes no coinciden para baseline vs modelo {model['name']}: {self.baseline.dims} vs {predictions.dims}"
                        ) from exc
            # Verifica que las fechas coincidan
            if not self._dates_match(predictions):
                raise ValueError(
                    f"""Las fechas no coinciden para el modelo {model['name']} 
                    (baseline: {self.baseline.datetime.values}, predictions: {predictions.datetime.values})
                    """
                    )
            # Si todo está bien, almacena los datos compatibles
            compatible_data_store.append(
                CompatibleData(model['name'], (predictions, self.baseline))
                )

        return compatible_data_store

    def _types_match(self, predictions: xr.Dataset) -> bool:
        """Verifica si los tipos de datos de las predicciones coinciden con los del baseline."""
        return type(self.baseline) == type(predictions)
    def _dims_size_match(self, predictions: xr.Dataset) -> bool:
        """Verifica si las dimensiones de las predicciones coinciden con las del baseline."""
        return len(self.baseline.dims) == len(predictions.dims)
    def _shapes_match(self, predictions: xr.Dataset) -> bool:
        """Verifica si los shapes de las predicciones coinciden con los del baseline."""
        return self.baseline.dims == predictions.dims
    def _lat_lon_match(self, predictions: xr.Dataset) -> bool:
        """Verifica si las latitudes y longitudes de las predicciones coinciden con las del baseline."""
        # Get the names of the lat and lon coordinates for the baseline
        self.lon_name_baseline = [key for key in self.baseline.coords.keys() if re.match(r'^lon', key)][0]
        self.lat_name_baseline = [key for key in self.baseline.coords.keys() if re.match(r'^lat', key)][0]
        # Get the names of the lat and lon coordinates for the predictions
        self.lon_name_pred = [key for key in predictions.coords.keys() if re.match(r'^lon', key)][0]
        self.lat_name_pred = [key for key in predictions.coords.keys() if re.match(r'^lat', key)][0]
        return (
            np.all(self.baseline.dims[self.lat_name_baseline] == predictions.dims[self.lat_name_pred])
            and np.all(self.baseline.dims[self.lon_name_baseline] == predictions.dims[self.lon_name_pred])
            )
    def _dates_match(self, predictions: xr.Dataset) -> bool:
        """Verifica si las fechas de las predicciones coinciden con las del baseline."""
        return np.all(self.baseline.datetime.values == predictions.datetime.values)


    @staticmethod
    def split_in_batch(sliced_dataset: xr.Dataset, n_lead_times: int) -> xr.Dataset:
        """Split the dataset into batches.

        Args:
        - sliced_dataset (xr.Dataset): The dataset to split.

        Returns:
        - xr.Dataset: The dataset split into batches.
        """

        window_size = n_lead_times #12
        WINDOW_STEP = 1 # 2
        IMPUT_DAYS = 2
        # Get the number of batches
        # window_size = sliced_dataset.time.size - (WINDOW_STEP * sample_size)
        batch_size = (sliced_dataset.time.size - window_size) / WINDOW_STEP
        batch_list = []
        for batch in np.arange(batch_size):
            # Get the step size
            step_size = int(WINDOW_STEP * batch if batch > 0 else 0)
            # Get the window
            window = np.array([0, window_size]) + step_size
            # Get the minimum reset
            min_reset = sliced_dataset.isel(
                time=slice(window[0], window[1])
                ).time.min()
            other_batch = sliced_dataset.isel(time=slice(window[0], window[1]))
            other_batch = other_batch.assign_coords(
                time=(other_batch.time - min_reset)
                )
            # assign datetime        (batch, time) datetime64[ns] coord
            other_batch = other_batch.assign_coords(
                datetime=(other_batch.time + min_reset)
                )
            batch_list.append(other_batch)

        return xr.concat(batch_list, dim='batch').isel(time=slice(None, window_size-IMPUT_DAYS))

    @staticmethod
    def downscaling(dataset: xr.Dataset, new_lonxlat: tuple) -> xr.Dataset:
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
