import numpy as np
import xarray as xr
import re
import functools
from copy import deepcopy
from art1_tools import interpolation

class ForecastEvaluator:
    """ForecastEvaluator class to evaluate the forecast performance.
    """
    def __init__(
        self, 
        forecast: xr.Dataset, 
        truth: xr.Dataset, 
        land_mask: xr.DataArray = None,
        weights_type: str = 'cos',
        climatology: xr.DataArray = None, 
        ) -> None:
        # Check if forecast and truth have the same shape
        ForecastEvaluator.check_shape(forecast, truth)
        # Create a land mask for the batched dataset
        self.land_mask = ForecastEvaluator.get_land_mask_batched(land_mask, truth)
        # Compute latitude weights based on weights_type 
        if weights_type == 'cos':
            self.lat_weights = LatitudeWeighter(truth).get_lat_weights_cos()
        elif weights_type == 'sin':
            self.lat_weights = LatitudeWeighter(truth).get_lat_weights_sin()
        else:
            raise ValueError("weights_type must be 'cos' or 'sin'.")
        # Define the forecast and truth datasets
        self.forecast = forecast
        self.truth = truth
        self.climatology = ForecastEvaluator.select_climatology_data(truth, climatology)


    def RMSE(self) -> xr.Dataset:
        """Compute the Root Mean Squared Error (RMSE) between the forecast and truth datasets.

        Computes RMSE latitude weighted for local domains. 
        Using the equation defined in: 
        Rasp, S., Hoyer, S., Merose, A., Langmore, I., Battaglia, P., 
        Russel, T., Sanchez-Gonzalez, A., Yang, V., Carver, R., Agrawal, S., Chantry, M., 
        Bouallegue, Z. B., Dueben, P., Bromberg, C., Sisk, J., Barrington, L., Bell, A., & Sha, F. (2024). 
        WeatherBench 2: A benchmark for the next generation of data-driven global weather models 
        (arXiv:2308.15560). arXiv. https://doi.org/10.48550/arXiv.2308.15560

        With the square root at the end.
        """
        diff_squared_fn = functools.partial(self.diff, squared=True)
        MSE = self.lat_weighted_agg(diff_squared_fn)
        MSE = MSE.mean('batch', skipna=True)
        caster = XarrayCaster(MSE)
        MSE: np.ndarray = caster.uncast_xarray()
        RMSE = MSE ** .5
        return caster.cast_xarray(RMSE) # Dimension: ('time')

    def RMSE_by_location(self) -> xr.Dataset:
        """Compute the spatial Root Mean Squared Error (RMSE) between the forecast and truth datasets.
        """
        square_diff = self.diff(squared=True)
        MSE = square_diff.mean('batch', skipna=True)
        # Extract sizes, coordinates and dims to cast results
        caster = XarrayCaster(MSE)
        MSE: np.ndarray = caster.uncast_xarray()
        RMSE = MSE ** .5

        return caster.cast_xarray(RMSE) # Dimension: ('time', 'lat', 'lon')

    def RMSE_by_batch(self) -> xr.Dataset:
        """Compute the Root Mean Squared Error (RMSE) between the forecast and truth datasets.
        """
        diff_squared_fn = functools.partial(self.diff, squared=True)
        MSE = self.lat_weighted_agg(diff_squared_fn)
        caster = XarrayCaster(MSE)
        MSE: np.ndarray = caster.uncast_xarray()
        RMSE = MSE ** .5

        return caster.cast_xarray(RMSE) # Dimension: ('batch', 'time')

    def bias(self) -> xr.Dataset:
        """Compute the bias between the forecast and truth datasets.
        """
        return self.bias_by_batch().mean('batch', skipna=True) # Dimension: ('time')

    def bias_by_location(self) -> xr.Dataset:
        """Compute the spatial bias between the forecast and truth datasets.
        """
        return self.diff().mean('batch', skipna=True) # Dimension: ('time', 'lat', 'lon')

    def bias_by_batch(self) -> xr.Dataset:
        """Compute the bias between the forecast and truth datasets.
        """
        return self.lat_weighted_agg(self.diff) # Dimension: ('batch', 'time')

    def activity(self) -> xr.Dataset:
        """Compute the activity between the forecast and truth datasets.
        """
        o_activity, f_activity = self.activity_by_batch()
        return o_activity.mean('batch', skipna=True), f_activity.mean('batch', skipna=True) # Dimension: ('time')

    def activity_by_batch(self) -> xr.Dataset:
        """Compute the activity between the forecast and truth datasets.
        Eq: sqrt( lat_weigh( ( (f - c ) - lat_weigh(f - c) )^2 ) )
        """
        # Compute the anomalies
        f_anomaly_mean = self.anomalies(self.forecast)
        o_anomaly_mean = self.anomalies(self.truth)
        o_anomaly_weighted = self.lat_weighted_agg(
            functools.partial(
                self.anomalies, 
                forecast_or_truth=self.truth
                )
            )
        f_anomaly_weighted = self.lat_weighted_agg(
            functools.partial(
                self.anomalies, 
                forecast_or_truth=self.forecast
                )
            )
        # Compute anomaly difference
        o_anomaly_diff = self.anomaly_diff(
            o_anomaly_mean,
            o_anomaly_weighted,
            )
        f_anomaly_diff = self.anomaly_diff(
            f_anomaly_mean,
            f_anomaly_weighted,
            )
        # Uncast dims and coords
        caster = XarrayCaster(f_anomaly_diff)
        _ = caster.uncast_xarray()
        caster_2 = caster.copy()


        f_wlat_square_diff = self.lat_weighted_square_diff(f_anomaly_diff, caster)
        o_wlat_square_diff = self.lat_weighted_square_diff(o_anomaly_diff, caster_2)


        return np.sqrt(o_wlat_square_diff), np.sqrt(f_wlat_square_diff) # Dimension: ('batch', 'time')
        
    def ACC(self) -> xr.Dataset:
        """Compute the anomaly correlation coefficient (ACC) between the forecast and truth datasets.
        """
        return self.ACC_by_batch().mean('batch', skipna=True) # Dimension: ('time')

    def ACC_by_batch(self) -> xr.Dataset:
        """Compute the anomaly correlation coefficient (ACC) between the forecast and truth datasets.
        lat_weigh( ((f - c) - lat_weigh(f - c)) * ((o - c) - lat_weigh(o - c)) ) 
        / ( √lat_weigh( ((f - c) - lat_weigh(f - c))^2 ) * √lat_weigh( ((o - c) - lat_weigh(o - c))^2 ) )
        """
        # Compute the anomalies
        f_anomaly_mean = self.anomalies(self.forecast)
        o_anomaly_mean = self.anomalies(self.truth)
        o_anomaly_weighted = self.lat_weighted_agg(
            functools.partial(
                self.anomalies, 
                forecast_or_truth=self.truth
                )
            )
        f_anomaly_weighted = self.lat_weighted_agg(
            functools.partial(
                self.anomalies, 
                forecast_or_truth=self.forecast
                )
            )
        # Compute anomaly difference
        o_anomaly_diff = self.anomaly_diff(
            o_anomaly_mean,
            o_anomaly_weighted,
            )
        f_anomaly_diff = self.anomaly_diff(
            f_anomaly_mean,
            f_anomaly_weighted,
            )
        # Compute the numerator and numerator of the ACC
        caster = XarrayCaster(o_anomaly_diff)
        _ = caster.uncast_xarray()
        caster_2 = caster.copy()
        caster_3 = caster.copy()
        ACC_numerator = self.ACC_numerator(f_anomaly_diff, o_anomaly_diff, caster)
        # Compute the denominator
        f_wlat_square_diff = self.lat_weighted_square_diff(f_anomaly_diff, caster_2)
        o_wlat_square_diff = self.lat_weighted_square_diff(o_anomaly_diff, caster_3)
        ACC_denominator = np.sqrt(f_wlat_square_diff) * np.sqrt(o_wlat_square_diff)
        # In the case of Reanalysis data, the time is unaligned with the other datasets
        if self.forecast.time.data[0] == np.timedelta64(0, 'D'):
            self.forecast['time'] = self.forecast['time'] + np.timedelta64(1, 'D')
      
        # Compute the ACC
        return ACC_numerator / ACC_denominator # Dimension: ('batch', 'time')


    def anomaly_diff(
            self, 
            anomaly_mean,
            anomaly_weighted,
            #o_anomaly_mean, 
            #f_anomaly_mean,
            #o_anomaly_weighted,
            #f_anomaly_weighted,
            ) -> xr.Dataset:
            """Compute the anomaly difference between the forecast and truth datasets.
            wheighed by the latitude: w_lat(x) = 
            Eq: anomaly_diff = (f - c) - (lat_weigh(f - c))
            """
            caster_anomaly_mean = XarrayCaster(anomaly_mean)
            caster_anomaly_weighted = XarrayCaster(anomaly_weighted)
            anomaly_mean: np.ndarray = caster_anomaly_mean.uncast_xarray()
            anomaly_weighted: np.ndarray = caster_anomaly_weighted.uncast_xarray()
            anomaly_diff = anomaly_mean - anomaly_weighted[:, :, np.newaxis, np.newaxis]

            return caster_anomaly_mean.cast_xarray(anomaly_diff)


    def ACC_numerator(self, f_anomaly_diff, o_anomaly_diff, caster):
        ACC_numerator = self.lat_weighted_agg(
            functools.partial(
                (lambda f_anomaly_diff, o_anomaly_diff: 
                    caster.cast_xarray(
                        f_anomaly_diff.data * o_anomaly_diff.data
                        )
                    ),
                f_anomaly_diff=f_anomaly_diff,
                o_anomaly_diff=o_anomaly_diff,
                )
            )
        return ACC_numerator

    # def ACC_denominator(self, anomaly_diff, caster):
    def lat_weighted_square_diff(self, anomaly_diff, caster):
        # Compute the denominator
        lat_weighted_square_diff = self.lat_weighted_agg(
            functools.partial(
                (lambda anomaly_diff:
                    caster.cast_xarray(anomaly_diff.data ** 2)),
                anomaly_diff=anomaly_diff,
                )
            )
        return lat_weighted_square_diff

    def lat_weighted_agg(self, fn) -> xr.Dataset:
        """Compute the latitude weighted bias between the forecast and truth datasets.
        """
        # Compute the bias
        bias = fn()
        # Get latitude and longitude keys
        lat_key = [
            key for key in bias.dims 
            if re.compile(r'^lat', re.IGNORECASE).match(key)
            ][0]
        lon_key = [
            key for key in bias.dims 
            if re.compile(r'^lon', re.IGNORECASE).match(key)
            ][0]
        # Compute the latitude weighted bias
        if self.lat_weights.dims[0] != lat_key:
            w_lat = self.lat_weights.rename(
                {self.lat_weights.dims[0]: lat_key}
                ).assign_coords(
                    {lat_key: bias[lat_key].data}
                    )
        else:
            w_lat = self.lat_weights
            
        bias_lat_weighted = (
            bias
            .weighted(w_lat)
            .mean([lat_key, lon_key], skipna=True)
            )
        
        return bias_lat_weighted

    def diff(self, squared: bool = False) -> xr.Dataset:
        """Compute the bias between the forecast and truth datasets.
        """
        # Extract sizes, coordinates and dims to cast results
        caster = XarrayCaster(self.truth)
        # Check if forecast and truth are numpy arrays
        forecast: np.ndarray = ForecastEvaluator.check_np_array(self.forecast)
        truth: np.ndarray = caster.uncast_xarray()
        #truth: np.ndarray = ForecastEvaluator.check_np_array(self.truth)
        # Compute the bias
        bias = (forecast - truth) ** 2 if squared else forecast - truth
        # Exclude land areas using landmask
        if self.land_mask is not None and self.land_mask.any():
            excluded_land = np.where(~self.land_mask.data, bias, np.nan)
            # Wrap excluded_land to xarray
            bias = caster.cast_xarray(excluded_land)
            # bias = xr.DataArray(excluded_land, dims=sizes, coords=coords)
        
        return bias
    
    def anomalies(self, forecast_or_truth: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
        """ Compute the anomalies between the target and climatology data.

        Eq1: anomaly f = f - c
        Eq2: anomaly o = o - c
        """    
        # Check the difference in resolution between target and climatology data
        assert ForecastEvaluator.check_tolerance(forecast_or_truth, self.climatology), \
            "The resolution difference between forecast_or_truth and climatology data exceeds the tolerance level."
        # Extract sizes, coordinates and dims to cast results
        caster = XarrayCaster(forecast_or_truth)
        forecast_or_truth: np.ndarray = caster.uncast_xarray()
        # Check if forecast and truth are numpy arrays
        forecast_or_truth: np.ndarray = ForecastEvaluator.check_np_array(forecast_or_truth)
        # Compute the anomaly between the target and climatology data
        anomaly: np.ndarray = forecast_or_truth - self.climatology.to_array().squeeze().data
        assert type(anomaly) == np.ndarray, "The anomaly must be a numpy DataArray."  
        # Transform the anomaly back to a xarray DataArray to keep the coordinates
        if self.land_mask is not None and self.land_mask.any():
            excluded_land = np.where(~self.land_mask.data, anomaly, np.nan)
            # Wrap excluded_land to xarray
            anomaly = caster.cast_xarray(excluded_land)
        
        return anomaly

    # Functio that checks if is a numpy array
    @staticmethod
    def check_np_array(array):
        """
        Check if the input is a numpy array or a xarray data array.
        """
        # Check if the array is a numpy array
        if isinstance(array, np.ndarray):
            return array
        elif isinstance(array, xr.DataArray):
            return array.data
        elif isinstance(array, xr.Dataset):
            raise ValueError("The input cannot be a xarray dataset")
        else:
            raise ValueError("The input is not a numpy array or a xarray data array")
    
    @staticmethod
    def get_land_mask_batched(data_nanland: xr.DataArray, ds_to_cast: xr.Dataset):
        """
        Get a land mask for a batched dataset.

        Parameters:
        - data_nanland (xr.DataArray): DataArray representing land areas (NaN values for land points).
        - ds_to_cast (xr.Dataset): Dataset containing the dimensions for which to create the land mask.

        Returns:
        - xr.DataArray: Land mask for the batched dataset.

        This function creates a land mask for a batched dataset based on a given land DataArray.
        It expands the land DataArray to match the batch dimension of the dataset and repeats it along this dimension.
        """
        # Check if ds_to_cast has dims equal to ('batch', 'time', 'lat', 'lon')
        assert ds_to_cast.dims == ('batch', 'time', 'lat', 'lon'), f"Dataset {ds_to_cast.dims} must have dimensions ('batch', 'time', 'lat', 'lon')."
        # Get the time size from the dataset
        time_size = ds_to_cast.sizes['time']
        # Create a boolean mask for land areas and expand it to match the time dimension
        bool_mask_time = np.expand_dims(np.isnan(data_nanland.data), axis=0)
        bool_mask_time = np.repeat(bool_mask_time, time_size, axis=0)
        # Get the batch size from the dataset
        batch_size = ds_to_cast.sizes['batch']
        # Create a boolean mask for land areas and expand it to match the batch dimension
        bool_mask_batch = np.expand_dims(bool_mask_time, axis=0)
        bool_mask_batch = np.repeat(bool_mask_batch, batch_size, axis=0)
        # Create a DataArray for the land mask with the same dimensions as the dataset
        land_mask = xr.DataArray(bool_mask_batch, dims=ds_to_cast.dims, coords=ds_to_cast.coords)

        return land_mask

    @staticmethod
    def check_shape(forecast: xr.Dataset, truth: xr.Dataset):
        """Check if the forecast and truth datasets have the same shape.
        """
        if isinstance(forecast, xr.DataArray):
            assert forecast.data.shape == truth.data.shape, \
                f"Forecast {forecast.data.shape} and truth {truth.data.shape} must have the same shape."
        else:
            raise ValueError("The input must be a xarray DataArray")

    # Define a function to check the tolerance of the resolution difference between target_arr and climatology data
    @staticmethod
    def check_tolerance(
        target_arr: xr.DataArray | xr.Dataset, 
        climatology_arr: xr.DataArray, 
        tol: float = 1e-3
        ) -> bool:
        """ This function checks if the resolution difference between the target and climatology data is within a specified tolerance level.

        target_arr : xr.DataArray or xr.Dataset
            The target data array or dataset containing time, latitude, and longitude dimensions.
        climatology_arr : xr.DataArray
            The climatology data array containing temperature potential data.
        tol : float
            The tolerance level for the resolution difference between the target and climatology data.
        
        note:
        -----
        The function gets the latitude of the target and climatology data, computes the resolution of the target and climatology data,
        and checks if the resolution difference between the target and climatology data is within the tolerance level.

        Returns
        -------
        bool
            True if the resolution difference between target and climatology data is within the tolerance level, False otherwise.

        """
        # Get the latitude of the target data
        lat_key = [key for key in target_arr.dims if re.compile(r'^lat', re.IGNORECASE).match(key)][0]
        target_lat = target_arr[lat_key]

        # Get resolution of target_arr data
        target_res = np.abs(np.unique(target_lat.diff(lat_key).data)).tolist()

        # Get the latitude of the climatology data
        lat_key = [key for key in climatology_arr.dims if re.compile(r'^lat', re.IGNORECASE).match(key)][0]
        clim_lat = climatology_arr[lat_key]

        # Get resolution of climatology data
        clim_res = np.abs(np.unique(clim_lat.diff(lat_key).data)).tolist()

        # Check if the resolution of target and climatology data are below the tolerance
        return np.abs(target_res[0] - clim_res[0]) < tol

    # Function to select climatology data for current month of target date
    @staticmethod
    def select_climatology_data(
        target: xr.DataArray | xr.Dataset, 
        climatology_data: xr.DataArray,
        ) -> xr.Dataset:
        """ Select climatology data for current month of target date

        Select climatology data for the current month of the target date.
        Parameters
        ----------
        target : xr.DataArray
            The target data array containing time, latitude, and longitude dimensions.
        climatology_data : xr.Dataset
            The climatology dataset containing temperature potential data.
        i_batch : int
            The batch index to select from the target data.
        i_time : int
            The time index to select from the target data.
        Returns
        -------
        xr.Dataset
            The selected climatology data for the current month of the target date.
        Raises
        ------
        AssertionError
            If the target time size is not equal to 10.
        Notes
        -----
        - The function removes the first two days of the target data.
        - The climatology data is resized to match the longitude and latitude dimensions of the target data.
        - The month is set as the time dimension in the climatology data.
        - The function selects the climatology data for the current month of the target date.
        """    
        # get lon and lat sizes
        lon_size = target.lon.size
        lat_size = target.lat.size

        assert isinstance(climatology_data, xr.Dataset), \
            "climatology_data must be a xr.Dataset not a xr.DataArray. Please do not select the variable first."
        # Resize climatology data
        
        climatology_data = interpolation.resize_lonxlat(
            climatology_data, 
            (lon_size, lat_size)
            )
        # Set month as time dimension in climatology data
        climatology_data["time"] = climatology_data["time.month"]
        # Convert from celsius to kelvin
        climatology_data = climatology_data + 273.15
        # Get month of current target day
        # assert target.time.size == 10, "target time size must be 10"

        # Define variable namew query to select the variable from the climatology data
        vars_query = {'analysed_sst': 'thetao'}
        target = target.transpose('batch', 'time', 'lat', 'lon')
        target = target.to_dataset()
        # Creatre an empty array to store the selected climatology data
        batch_size = target.batch.size
        time_size = target.time.size
        selected_clim = {}

        for var_name in target.data_vars:
            selected_clim[var_name] = np.zeros((batch_size, time_size, lat_size, lon_size))

            for batch in range(target.batch.size):
                for time in range(target.time.size):
                    current_date = (
                        target
                        .datetime.dt.month
                        .isel(batch=batch)
                        .isel(time=time)
                        )
                    # Select climatology data for current month of target date
                    try:    
                        selected_clim[var_name][batch, time, :, :] = (
                            climatology_data[vars_query[var_name]]
                            .sel(time=current_date)
                            )
                    except KeyError:
                        print(f"Variable {var_name} not found in climatology data.")
                        break

        climatology_dataset = xr.Dataset(
            {var_name: (target['analysed_sst'].dims, xarr) for var_name, xarr in selected_clim.items()},
            coords=target['analysed_sst'].coords
            )
        return climatology_dataset

def RMSE(
    forecast: xr.Dataset, 
    truth: xr.Dataset, 
    land_mask: xr.DataArray,
    weights_type: str = 'cos', 
    skipna: bool = True,
    along_batch: bool = True,
    ) -> xr.Dataset:
    """
    Computes RMSE latitude weighted for local domains. 
    Using the equation defined in: 
    Rasp, S., Hoyer, S., Merose, A., Langmore, I., Battaglia, P., 
    Russel, T., Sanchez-Gonzalez, A., Yang, V., Carver, R., Agrawal, S., Chantry, M., 
    Bouallegue, Z. B., Dueben, P., Bromberg, C., Sisk, J., Barrington, L., Bell, A., & Sha, F. (2024). 
    WeatherBench 2: A benchmark for the next generation of data-driven global weather models 
    (arXiv:2308.15560). arXiv. https://doi.org/10.48550/arXiv.2308.15560

    """
    # Check if forecas and truth have the same shape
    assert forecast.data.shape == truth.data.shape, f"Forecast {forecast.data.shape} and truth {truth.data.shape} must have the same shape."
    # Extract sizes, coordinates and dims to cast results
    sizes, coords = truth.sizes, truth.coords
    # Check if forecast and truth are numpy arrays
    forecast: np.ndarray = ForecastEvaluator.check_np_array(forecast)
    truth: np.ndarray = ForecastEvaluator.check_np_array(truth)
    land_mask: np.ndarray = ForecastEvaluator.check_np_array(land_mask)
    # Compute squared errors
    SE = (forecast - truth) ** 2
    # Exclude land areas using landmask
    if land_mask is not None and land_mask.any():
        excluded_land = np.where(~land_mask, SE, np.nan)
        SE = xr.DataArray(excluded_land, dims=sizes, coords=coords)
    # Get latitude and longitude keys
    lat_key = [key for key in SE.dims if re.compile(r'^lat', re.IGNORECASE).match(key)][0]
    lon_key = [key for key in SE.dims if re.compile(r'^lon', re.IGNORECASE).match(key)][0]
    if along_batch:
        avr_dims = [lat_key, lon_key, 'batch'] if 'batch' in SE.dims else [lat_key, lon_key]
    else:
        avr_dims = [lat_key, lon_key]
    # Compute weights based on weights_type
    match weights_type:
        case 'cos':
            weights = LatitudeWeighter(SE).get_lat_weights_cos()
            wlat_MSE = SE.weighted(weights).mean(avr_dims, skipna=skipna)
        case 'sin':
            weights = LatitudeWeighter(SE).get_lat_weights_sin()
            wlat_MSE = SE.weighted(weights).mean(avr_dims, skipna=skipna)
        case None:
            wlat_MSE = SE#.mean(avr_dims, skipna=skipna) # Not weighted
            
    return wlat_MSE ** .5


def RMSE_mean_std(
    forecast: xr.Dataset, 
    truth: xr.Dataset, 
    land_mask: xr.DataArray,
    weights_type: str = 'cos', 
    skipna: bool = True,
    along_batch: bool = True,
    ) -> tuple[xr.Dataset]:
    """
    Computes the Root Mean Squared Error (RMSE) and its standard deviation.

    Parameters:
    - forecast (xr.Dataset): Dataset containing forecast data.
    - truth (xr.Dataset): Dataset containing truth data.
    - landmask (xr.DataArray): 2D DataArray representing land mask.
    - weights_type (str, optional): Type of weights to apply ('cos' or 'sin'). Defaults to 'cos'.
    - skipna (bool, optional): Whether to exclude NaN values when computing mean and std. Defaults to True.

    Returns:
    - tuple: Tuple containing the RMSE mean and its standard deviation as DataArrays.
    """
    # Check if forecas and truth have the same shape
    assert forecast.data.shape == truth.data.shape, f"Forecast {forecast.data.shape} and truth {truth.data.shape} must have the same shape."
    # Compute squared errors
    SE = (forecast.data - truth.data) ** 2
    # Exclude land areas using landmask
    # Check if SE is a JaxArrayWrapper type:
    if not isinstance(SE, np.ndarray):
        SE = SE.jax_array

    excluded_land = np.where(~land_mask.data, SE, np.nan)
    # Wrap excluded_land to xarray
    excluded_land = xr.DataArray(
        excluded_land, 
        dims=truth.sizes, 
        coords=truth.coords,
        )
    # Compute weights based on weights_type
    match weights_type:
        case 'cos':
            weights = get_lat_weights_cos(excluded_land)
            wlat_MSE = excluded_land.weighted(weights)
        case 'sin':
            weights = get_lat_weights_sin(excluded_land)
            wlat_MSE = excluded_land.weighted(weights)
        case None:
            wlat_MSE = excluded_land # Not weighted

    # Compute RMSE mean and standard deviation along the 'batch' dimension if present          
    if along_batch:
        wlat_MSE_mean_B = (
            wlat_MSE.mean(['batch'], skipna=skipna) 
            if 'batch' in truth.dims else wlat_MSE
            )  
        wlat_MSE_Bstd_B = (
            wlat_MSE.std(['batch'], skipna=skipna) 
            if 'batch' in truth.dims else wlat_MSE
            )  

        return wlat_MSE_mean_B.squeeze() ** .5, wlat_MSE_Bstd_B.squeeze()
    else:
        return wlat_MSE, None

def get_land_mask_batched(data_nanland: xr.DataArray, ds_to_cast: xr.Dataset):
    """
    Get a land mask for a batched dataset.

    Parameters:
    - data_nanland (xr.DataArray): DataArray representing land areas (NaN values for land points).
    - ds_to_cast (xr.Dataset): Dataset containing the dimensions for which to create the land mask.

    Returns:
    - xr.DataArray: Land mask for the batched dataset.

    This function creates a land mask for a batched dataset based on a given land DataArray.
    It expands the land DataArray to match the batch dimension of the dataset and repeats it along this dimension.
    """
    # Check if ds_to_cast has dims equal to ('batch', 'time', 'lat', 'lon')
    assert ds_to_cast.dims == ('batch', 'time', 'lat', 'lon'), f"Dataset {ds_to_cast.dims} must have dimensions ('batch', 'time', 'lat', 'lon')."
    # Get the time size from the dataset
    time_size = ds_to_cast.sizes['time']
    # Create a boolean mask for land areas and expand it to match the time dimension
    bool_mask_time = np.expand_dims(np.isnan(data_nanland).data, axis=0)
    bool_mask_time = np.repeat(bool_mask_time, time_size, axis=0)
    # Get the batch size from the dataset
    batch_size = ds_to_cast.sizes['batch']
    # Create a boolean mask for land areas and expand it to match the batch dimension
    bool_mask_batch = np.expand_dims(bool_mask_time, axis=0)
    bool_mask_batch = np.repeat(bool_mask_batch, batch_size, axis=0)
    # Create a DataArray for the land mask with the same dimensions as the dataset
    land_mask = xr.DataArray(bool_mask_batch, dims=ds_to_cast.dims, coords=ds_to_cast.coords)

    return land_mask


def cast_forecast_truth(
    predictions: xr.Dataset, 
    example_batch: xr.Dataset,  
    target_vars_name: list[str] or str
    ) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Cast forecast and truth datasets to the same format.
    """
    target_vars_name = target_vars_name
    delta_t_1 = (example_batch.isel(time=-1)['time'].data 
                 - example_batch.isel(time=-2)['time'].data) 
    truth = (example_batch.isel(time=slice(2, 3))[target_vars_name].to_dataset()
                                .rename({'lat': 'latitude', 'lon': 'longitude'}))
    truth['time'] = [delta_t_1]
    forecast = (predictions.rename({'lat': 'latitude', 'lon': 'longitude'})
                           .assign_coords(datetime=(('batch', 'time'), 
                                                    truth['datetime'].data)))
    return (forecast, truth)

def dummy_ds(
    varname: str, 
    batch_size: int, 
    time_size: int,
    lat_size: int, 
    lon_size: int, 
    max_lon: float, 
    min_lon: float, 
    max_lat: float, 
    min_lat: float
    ) -> xr.Dataset:
    data = np.random.rand(batch_size, time_size, lat_size, lon_size)

    # Create the xarray DataArray
    dummy_da = xr.DataArray(
        data, 
        dims=['batch', 'time', 'lat', 'lon'], 
        coords={
            'batch': range(batch_size), 
            'time': np.arange(
                np.timedelta64(0, 'D'), 
                np.timedelta64(time_size, 'D')
                ),
            # 'datetime': (
            #     'time', 
            #     np.arange(
            #         np.datetime64('2021-01-01'), 
            #         np.datetime64(f'2021-01-{str((batch_size * time_size)+1).zfill(2)}')
            #         ).reshape(batch_size, time_size)
            #     ),                    
            'lat': np.linspace(min_lat, max_lat, lat_size), 
            'lon': np.linspace(min_lon, max_lon, lon_size),
            }
        )

    # Create the xarray Dataset
    return xr.Dataset({varname: dummy_da})

def ACC(
    f: xr.DataArray,
    o: xr.DataArray,
    c: xr.DataArray,
    )-> xr.DataArray:
    """ Compute the anomaly correlation coefficient (ACC) between the target and prediction data.

    We use the ACC formula difined in:
    Ben-Bouallegue, Z., Clare, M. C. A., Magnusson, L., Gascon, E., 
    Maier-Gerber, M., Janousek, M., Rodwell, M., Pinault, F., Dramsch, J. S., 
    Lang, S. T. K., Raoult, B., Rabier, F., Chevallier, M., Sandu, I., Dueben, P., 
    Chantry, M., & Pappenberger, F. (2023). 
    The rise of data-driven weather forecasting (arXiv:2307.10128). arXiv. 
    https://doi.org/10.48550/arXiv.2307.10128

    The ACC is defined as:
    $$
        \frac{\overline{(f - c - \overline{f - c})(o - c - \overline{o - c})}}{\sqrt{\overline{(f - c - \overline{f - c})^2} \, \overline{(o - c - \overline{o - c})^2}}},
    $$

    Parameters
    ----------
    f : xr.DataArray
        The prediction data array containing time, latitude, and longitude dimensions.
    o : xr.DataArray
        The target data array containing time, latitude, and longitude dimensions.
    c : xr.DataArray
        The climatology data array containing temperature potential data.

    Returns
    -------
    xr.DataArray
        The anomaly correlation coefficient between the target and prediction data.

    Raises
    ------
    AssertionError
        If the time dimension size of the target data is not equal to 10.
        If the resolution difference between target and climatology data exceeds the tolerance level.
        If the computed anomaly is not a numpy DataArray.

    Notes
    -----
    - The function removes the first two days of the target data.
    - The climatology data is resized to match the longitude and latitude dimensions of the target data.
    - The month is set as the time dimension in the climatology data.
    - The function selects the climatology data for the current month of the target date.
    - The function computes the anomalies between the target and climatology data.
    - The function computes the mean and weighted mean of the anomalies.
    - The function computes the anomaly correlation coefficient between the target and prediction data.
    """    
    # Select the forecasted days of the target data
    o = o.isel(time=slice(2, None))
    # Compute the anomalies between the target and climatology data
    o_anomaly_mean, o_anomaly_weighted = compute_anomalies(o, c)
    f_anomaly_mean, f_anomaly_weighted = compute_anomalies(f, c)

    # Convert anomalies to numpy arrays
    o_anomaly_mean_arr = o_anomaly_mean.data
    o_anomaly_weighted_arr = o_anomaly_weighted.data[:, :, np.newaxis, np.newaxis]
    f_anomaly_mean_arr = f_anomaly_mean.data
    f_anomaly_weighted_arr = f_anomaly_weighted.data[:, :, np.newaxis, np.newaxis]

    # NUMERATOR
    # Compute the numerator and denominator 
    arr_numerator = (
        (f_anomaly_mean_arr - f_anomaly_weighted_arr)
        * (o_anomaly_mean_arr - o_anomaly_weighted_arr)
        )
    # Wrapping to xarray again
    xarr_numerator = xr.DataArray(
        arr_numerator, 
        dims=o_anomaly_mean.dims,
        coords=o_anomaly_mean.coords
        )
    # Compute latitude weights
    lat_weights = get_lat_weights_cos(xarr_numerator)
    # Compute the weighted mean of the numerator
    lat_key = [key for key in xarr_numerator.dims if re.compile(r'^lat', re.IGNORECASE).match(key)][0]
    lon_key = [key for key in xarr_numerator.dims if re.compile(r'^lon', re.IGNORECASE).match(key)][0]
    numerator_weighted = xarr_numerator.weighted(lat_weights).mean(dim=[lat_key, lon_key], skipna=True)
    # Convert back to numpy array
    arr_numerator_weighted = numerator_weighted.data
    # DENOMINATOR
    # Compute the left and right side of denominator
    left_side_denominator = (f_anomaly_mean_arr - f_anomaly_weighted_arr)**2
    right_side_denominator = (o_anomaly_mean_arr - o_anomaly_weighted_arr)**2
    # Wrapping to xarray again
    xarr_left_denominator = xr.DataArray(
        left_side_denominator, 
        dims=o_anomaly_mean.dims,
        coords=o_anomaly_mean.coords
        )
    xarr_right_denominator = xr.DataArray(
        right_side_denominator, 
        dims=o_anomaly_mean.dims,
        coords=o_anomaly_mean.coords
        )
    # Compute the weighted mean of left and right side of the denominator
    left_denominator_weighted = xarr_left_denominator.weighted(lat_weights).mean(dim=[lat_key, lon_key], skipna=True)
    right_denominator_weighted = xarr_right_denominator.weighted(lat_weights).mean(dim=[lat_key, lon_key], skipna=True)
    # Convert back to numpy arrays
    left_denominator_weighted_arr = left_denominator_weighted.data
    right_denominator_weighted_arr = right_denominator_weighted.data
    # Compute the denominator
    arr_denominator_rooted = np.sqrt(
        left_denominator_weighted_arr 
        * right_denominator_weighted_arr
        )
    # Compute the final correlation
    acc = arr_numerator_weighted / arr_denominator_rooted
    # Wrap the final correlation to xarray
    xarr_acc = xr.DataArray(
        acc, 
        dims=('batch', 'time'), 
        coords={'batch': o_anomaly_mean.batch, 'time': o_anomaly_mean.time}
        )
    # Define the initial date of the batch
    init_dates = o.datetime.isel(time=1).data
    # Replace the batch dim by a batch_initial_date dim
    xarr_acc = xarr_acc.rename({'batch': 'initial_date'})
    xarr_acc['initial_date'] = init_dates
    
    return xarr_acc
    
# Function to select climatology data for current month of target date
def select_climatology_data(
    target: xr.DataArray | xr.Dataset, 
    climatology_data: xr.DataArray,
    ) -> xr.Dataset:
    """ Select climatology data for current month of target date

    Select climatology data for the current month of the target date.
    Parameters
    ----------
    target : xr.DataArray
        The target data array containing time, latitude, and longitude dimensions.
    climatology_data : xr.Dataset
        The climatology dataset containing temperature potential data.
    i_batch : int
        The batch index to select from the target data.
    i_time : int
        The time index to select from the target data.
    Returns
    -------
    xr.Dataset
        The selected climatology data for the current month of the target date.
    Raises
    ------
    AssertionError
        If the target time size is not equal to 10.
    Notes
    -----
    - The function removes the first two days of the target data.
    - The climatology data is resized to match the longitude and latitude dimensions of the target data.
    - The month is set as the time dimension in the climatology data.
    - The function selects the climatology data for the current month of the target date.
    """    
    # Remove first two days of target data
    target = target.isel(time=slice(2, None))

    # get lon and lat sizes
    lon_size = target.lon.size
    lat_size = target.lat.size

    assert isinstance(climatology_data, xr.Dataset), "climatology_data must be a xr.Dataset not a xr.DataArray. Select the variable first."
    # Resize climatology data
    climatology_data = interpolation.resize_lonxlat(
        climatology_data, 
        (lon_size, lat_size)
        )
    # Set month as time dimension in climatology data
    climatology_data["time"] = climatology_data["time.month"]
    # Get month of current target day
    assert target.time.size == 10, "target time size must be 10"

    # Define variable namew query to select the variable from the climatology data
    vars_query = {'analysed_sst': 'thetao'}

    # Creatre an empty array to store the selected climatology data
    batch_size = target.batch.size
    time_size = target.time.size
    selected_clim = {}

    for var_name in target.data_vars:
        selected_clim[var_name] = np.zeros((batch_size, time_size, lat_size, lon_size))

        for batch in range(target.batch.size):
            for time in range(target.time.size):
                current_date = (
                    target
                    .datetime.dt.month
                    .isel(batch=batch)
                    .isel(time=time)
                    )
                # Select climatology data for current month of target date
                try:    
                    selected_clim[var_name][batch, time, :, :] = (
                        climatology_data[vars_query[var_name]]
                        .sel(time=current_date)
                        )
                except KeyError:
                    print(f"Variable {var_name} not found in climatology data.")
                    break

    climatology_dataset = xr.Dataset(
        {var_name: (target.dims, xarr) for var_name, xarr in selected_clim.items()},
        coords=target.coords
        )
    return climatology_dataset

# Define a function to check the tolerance of the resolution difference between target_arr and climatology data
def check_tolerance(
    target_arr: xr.DataArray | xr.Dataset, 
    climatology_arr: xr.DataArray, 
    tol: float = 1e-3
    ) -> bool:
    """ This function checks if the resolution difference between the target and climatology data is within a specified tolerance level.

    target_arr : xr.DataArray or xr.Dataset
        The target data array or dataset containing time, latitude, and longitude dimensions.
    climatology_arr : xr.DataArray
        The climatology data array containing temperature potential data.
    tol : float
        The tolerance level for the resolution difference between the target and climatology data.
    
    note:
    -----
    The function gets the latitude of the target and climatology data, computes the resolution of the target and climatology data,
    and checks if the resolution difference between the target and climatology data is within the tolerance level.

    Returns
    -------
    bool
        True if the resolution difference between target and climatology data is within the tolerance level, False otherwise.

    """
    # Get the latitude of the target data
    lat_key = [key for key in target_arr.dims if re.compile(r'^lat', re.IGNORECASE).match(key)][0]
    target_lat = target_arr[lat_key]

    # Get resolution of target_arr data
    target_res = np.abs(np.unique(target_lat.diff(lat_key).data)).tolist()

    # Get the latitude of the climatology data
    lat_key = [key for key in climatology_arr.dims if re.compile(r'^lat', re.IGNORECASE).match(key)][0]
    clim_lat = climatology_arr[lat_key]

    # Get resolution of climatology data
    clim_res = np.abs(np.unique(clim_lat.diff(lat_key).data)).tolist()

    # Check if the resolution of target and climatology data are below the tolerance
    return np.abs(target_res[0] - clim_res[0]) < tol

# Select climatology data for current month of target date
def compute_anomalies(
    target: xr.DataArray, 
    climatology_data: xr.Dataset
    ) -> tuple[xr.DataArray, xr.DataArray]:
    """ Compute the anomalies between the target and climatology data.

    Parameters
    ----------
    target : xr.DataArray
        The target data array containing time, latitude, and longitude dimensions.
    climatology_data : xr.Dataset
        The climatology dataset containing temperature potential data.

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        The mean anomalies and weighted mean anomalies between the target and climatology data.

    Raises
    ------
    AssertionError
        If the target time size is not equal to 10.
        If the resolution difference between target and climatology data exceeds the tolerance level.
        If the computed anomaly is not a numpy DataArray.

    Notes
    -----
    - The function removes the first two days of the target data.
    - The climatology data is resized to match the longitude and latitude dimensions of the target data.
    - The month is set as the time dimension in the climatology data.
    - The function selects the climatology data for the current month of the target date.
    - The function computes the anomalies between the target and climatology data.
    - The function computes the mean and weighted mean of the anomalies.
    """    
    # Check if time dimension size of target data is equal to 10
    assert target.time.size == 10, "The target time size must be equal to 10."
    # Check the difference in resolution between target and climatology data
    assert check_tolerance(target, climatology_data), "The resolution difference between target and climatology data exceeds the tolerance level."

    # Compute the anomaly between the target and climatology data
    anomaly: np.ndarray = target.data - climatology_data.data
    assert type(anomaly) == np.ndarray, "The anomaly must be a numpy DataArray."  
    # Transform the anomaly back to a xarray DataArray to keep the coordinates
    anomaly_arr = xr.DataArray(anomaly, dims=target.dims, coords=target.coords)

    # Get latitude weights
    lat_weights = get_lat_weights_cos(anomaly_arr)
    # Map reduce the anomaly target data
    lat_key = [key for key in anomaly_arr.dims if re.compile(r'^lat', re.IGNORECASE).match(key)][0]
    lon_key = [key for key in anomaly_arr.dims if re.compile(r'^lon', re.IGNORECASE).match(key)][0]
    anomaly_mean = anomaly_arr

    # Compute the weighted mean of the anomaly target data
    anomaly_weighted = anomaly_arr.weighted(lat_weights).mean(dim=[lat_key, lon_key], skipna=True)
    
    return anomaly_mean, anomaly_weighted




class LatitudeWeighter:
    """LatitudeWeighter class to compute latitude/area weights from latitude coordinate of dataset.
    """
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    def _latitude_cell_bounds(self, x: np.ndarray) -> np.ndarray:
        """Calculate the latitude cell bounds."""
        pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
        delta1 = (x[0] + x[1]) / 2
        delta2 = (x[-1] + x[-2]) / 2
        return np.concatenate([x[0:1] - (delta1 - x[0:1]), (x[:-1] + x[1:]) / 2, x[-1:] + (x[-1:] - delta2)])

    def _cell_area_from_latitude(self, points: np.ndarray) -> np.ndarray:
        """Calculate the area overlap as a function of latitude."""
        bounds = self._latitude_cell_bounds(points)
        # _assert_increasing(bounds)
        upper = bounds[1:]
        lower = bounds[:-1]
        # normalized cell area: integral from lower to upper of cos(latitude)
        return np.sin(upper) - np.sin(lower)

    def get_lat_weights_sin(self) -> xr.DataArray:
        """
        Computes latitude/area weights from latitude coordinate of dataset. 
        Using the equation defined in: 
        Rasp, S., Hoyer, S., Merose, A., Langmore, I., Battaglia, P., 
        Russel, T., Sanchez-Gonzalez, A., Yang, V., Carver, R., Agrawal, S., Chantry, M., 
        Bouallegue, Z. B., Dueben, P., Bromberg, C., Sisk, J., Barrington, L., Bell, A., & Sha, F. (2024). 
        WeatherBench 2: A benchmark for the next generation of data-driven global weather models 
        (arXiv:2308.15560). arXiv. https://doi.org/10.48550/arXiv.2308.15560
        """
        ds: xr.Dataset = self.ds
        lat_key = [key for key in ds.dims if re.compile(r'^lat', re.IGNORECASE).match(key)][0]
        weights = self._cell_area_from_latitude(np.deg2rad(ds[lat_key].data))
        weights /= np.mean(weights)
        weights = ds[lat_key].copy(data=weights)
        return weights

    def get_lat_weights_cos(self) -> xr.DataArray:
        """
        Computes latitude/area weights from latitude coordinate of dataset. 
        Using the equation defined in: 
        Rasp, S., Dueben, P. D., Scher, S., Weyn, J. A., Mouatadid, S., & Thuerey, N. (2020). 
        WeatherBench: A benchmark dataset for data-driven weather forecasting. 
        Journal of Advances in Modeling Earth Systems, 12(11), e2020MS002203. 
        https://doi.org/10.1029/2020MS002203

        > > $$w(j)=\frac{\cos (lat_j)}{\frac{1}{N_{lat }} \sum_j^{N_{\text {lat }}} \cos (lat_j)}$$
        
        """
        ds: xr.Dataset = self.ds
        # Get latitude key
        lat_key = [key for key in ds.dims if re.compile(r'^lat', re.IGNORECASE).match(key)][0]
        # Compute latitude weights
        weights = np.cos(np.deg2rad(ds[lat_key]))
        # Divide by the mean of the cosine of the latitude 
        # This could seam like a normalization, but it is actually a weighting
        # This mean is then cancel with 
        weights /= weights.mean(dim=[lat_key], skipna=False)

        return weights


class XarrayCaster:
    """Class to cast an xarray DataArray to a numpy array and viceversa"""

    def __init__(self, xarr: xr.DataArray):
        self.xarr = xarr
        self.sizes = None 
        self.coords = None
    
    def uncast_xarray(self) -> np.ndarray:
        self.sizes, self.coords = self.xarr.sizes, self.xarr.coords

        return self.xarr.data

    def cast_xarray(self, new_xarr: np.ndarray) -> xr.DataArray:
        self.xarr = new_xarr
        self.xarr = xr.DataArray(self.xarr, dims=self.sizes, coords=self.coords)
        self.sizes, self.coords = None, None

        return self.xarr

    def copy(self):
        """Create a deep copy of the XarrayCaster instance.
        
        Returns:
            XarrayCaster: A new instance with copied data
        """
        new_instance = XarrayCaster(deepcopy(self.xarr))
        new_instance.sizes = deepcopy(self.sizes)
        new_instance.coords = deepcopy(self.coords)
        return new_instance