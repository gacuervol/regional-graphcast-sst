import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import itertools
from google.cloud import storage
from scipy.ndimage import gaussian_filter
from . import interpolation
from . import replace

# from . import utils_data_loader

def get_username(path_list):
    home_index = path_list.index('home')
    user = path_list[home_index + 1]
    return user 

class Loader:
    
    def __init__(
        self, 
        n_lead_times: int, 
        min_batch_size: int,
        data_path: str,
        free_memory: bool = True,
        ) -> None:

        self.n_lead_times = n_lead_times
        self.min_batch_size = min_batch_size
        self.data_path = data_path
        self.free_memory = free_memory
        # Load local data
        #path_export_data = "
        # Preprocess the data to match the GraphCast data
        self.example_batch = self.prepare_dataset(
            xr.load_dataset(self.data_path) #+ 'IBI_SST_L4_FULL.nc'#"IBI_SST_L4_2016_2021.nc"
            )
        # Store a slice of the data with nan values
        self.SST_SAT_with_nan = xr.load_dataset(
            self.data_path
            ).isel(time=0) # Data for spatial weights

    def __call__(self) -> tuple[iter, iter, iter]:
        
        n_lead_times = self.n_lead_times
        example_batch = self.example_batch
        # instance of the DataSplitter class
        data_splitter = DataSplitter(
            example_batch, 
            train_size=0.8, 
            validation_size=0.1,
            )
        # Split the data into train, validation and test sets
        train_set, val_set, test_set = data_splitter.split_data()
        # Remove example_batch to free memory
        if self.free_memory:
            del self.example_batch
        # Store time range
        self.train_range = (
            str(train_set.datetime.isel(time=0).squeeze().data),
            str(train_set.datetime.isel(time=-1).squeeze().data)
            )
        self.val_range = (
            str(val_set.datetime.isel(time=0).squeeze().data),
            str(val_set.datetime.isel(time=-1).squeeze().data)
            )
        self.test_range = (
            str(test_set.datetime.isel(time=0).squeeze().data),
            str(test_set.datetime.isel(time=-1).squeeze().data)
            )
        # Store the train, validation and test sets
        self.train_example_batch = train_set
        self.val_example_batch = val_set
        self.test_example_batch = test_set
        # Cyclic dataset generators
        min_batch_size = self.min_batch_size
        gentype = 'linear'
        # Create train batched generator
        train_batched_gen = BatchGenerator(
            self.train_example_batch, 
            min_batch_size=min_batch_size, 
            sliding_window=n_lead_times,
            gentype=gentype,
            shuffle=True,
            ).generate()
        # Remove train_set to free memory
        if self.free_memory:
            del self.train_example_batch
        # Create validation batched generator
        val_batched_gen = BatchGenerator(
            self.val_example_batch, 
            min_batch_size=min_batch_size, 
            sliding_window=n_lead_times,
            gentype=gentype,
            shuffle=False,
            ).generate()
        # Remove val_set to free memory
        # if self.free_memory:
        del self.val_example_batch
        # Create test batched generator
        test_batched_gen = BatchGenerator(
            self.test_example_batch, 
            min_batch_size=min_batch_size, 
            sliding_window=n_lead_times,
            gentype=gentype,
            shuffle=False,
            ).generate()
        # Remove test_set to free memory
        # if self.free_memory:
        del self.test_example_batch
        
        return train_batched_gen, val_batched_gen, test_batched_gen 

    def load_norms(self) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        # Preprocess the data to match the GraphCast data
        example_batch = self.prepare_dataset(
            xr.load_dataset(self.data_path) #+ 'IBI_SST_L4_FULL.nc'#"IBI_SST_L4_2016_2021.nc"
            )
        # Cliente Google Cloud Storage anónimo.
        gcs_client = storage.Client.create_anonymous_client()

        # Obtener el bucket "dm_graphcast" del cliente de Google Cloud Storage.
        gcs_bucket = gcs_client.get_bucket("dm_graphcast")

        # Load norm matrices Cargar los datos de diferencias y desviaciones estándar por nivel.
        with gcs_bucket.blob("stats/diffs_stddev_by_level.nc").open("rb") as f:
            diffs_stddev_by_level = xr.load_dataset(f).compute()

        # Cargar los datos de media por nivel.
        with gcs_bucket.blob("stats/mean_by_level.nc").open("rb") as f:
            mean_by_level = xr.load_dataset(f).compute()

        # Cargar los datos de desviación estándar por nivel.
        with gcs_bucket.blob("stats/stddev_by_level.nc").open("rb") as f:
            stddev_by_level = xr.load_dataset(f).compute()

        # Get container folder
        containing_folder_path = os.path.dirname(self.data_path)
        # folder = 
        #dir_data_linux = 
        # Load norm matrices from SST_SAT
        SST_SAT_mean_1982_2015 = xr.load_dataset(
            containing_folder_path + "/mean_sat_1982_2012.nc" 
            )
        SST_SAT_std_1982_2015 = xr.load_dataset(
            containing_folder_path + "/std_sat_1982_2012.nc" 
            )
        SST_SAT_diff_std_1982_2015 = xr.load_dataset(
            containing_folder_path + "/diff_std_sat_1982_2012.nc" 
            )

        # Replace for our nonrm data
        # diffs_stddev_by_level = replace_norm_drops_vars(
        #     diffs_stddev_by_level,
        #     SST_SAT_diff_std_1982_2015["analysed_sst"],
        #     example_batch,
        #     "analysed_sst",
        #     )
        # mean_by_level = replace_norm_drops_vars(
        #     mean_by_level,
        #     SST_SAT_mean_1982_2015["analysed_sst"],
        #     example_batch,
        #     "analysed_sst",
        #     )
        # stddev_by_level = replace_norm_drops_vars(
        #     stddev_by_level,
        #     SST_SAT_std_1982_2015["analysed_sst"],
        #     example_batch,
        #     "analysed_sst",
        #     )
        diffs_stddev_by_level = SST_SAT_diff_std_1982_2015
        mean_by_level = SST_SAT_mean_1982_2015
        stddev_by_level = SST_SAT_std_1982_2015
        
        return mean_by_level, stddev_by_level, diffs_stddev_by_level

    def prepare_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        dataset = self.adapt_coords_ds(dataset)
        land_sea_mask: np.ndarray = Loader.generate_land_sea_mask(dataset)
        # Get dimensions names that start with 'lat' or 'lon'
        lat_lon_dims = [
            dim for dim in dataset.dims 
            if dim.startswith('lat') or dim.startswith('lon')
            ]
        # Assign the land-sea mask as new variable
        dataset['land_sea_mask'] = (lat_lon_dims, land_sea_mask)

        return dataset

    def adapt_coords_ds(self, dataset: xr.Dataset) -> xr.Dataset:
        dates = dataset.time.data
        # add 12 hours to the time coordinate to get day time solar radiation
        # Check if all dates are at midnight
        if all(dates.astype('datetime64[h]') == dates.astype('datetime64[D]')):
            dataset['time'] = dates + np.timedelta64(12, 'h') # add 12 hours
            dates = dataset.time.data
        # Add a batch dimension
        dataset = (
            dataset.expand_dims('batch')
            .assign_coords(datetime=(('batch', 'time'), [dates]))
            )
        # Create a vector starting from 0 with the same length as dates
        indices = np.arange(len(dataset.time.data))
        time_coord_data =  np.insert(
            ((dates[1:] - dates[0:-1]) * indices[1:]), 
            0, 
            0,
            )
        dataset = dataset.assign_coords(time=time_coord_data)
        dataset = dataset.rename(
            {'longitude': 'lon', 'latitude': 'lat'}
            )
        
        return dataset 

    # Define a function to generate land-sea mask from SST_SAT
    @staticmethod
    def generate_land_sea_mask(
        dataset: xr.Dataset, 
        sigma: float = 1.5,
        ) -> xr.Dataset:
        # Get variables names
        dataset_vars = list(dataset.data_vars)
        # Get the first variable
        data_arr = dataset[dataset_vars[0]]
        # Check if time is in the coordinates
        # Select the first time and batch dimension if they exist
        for coord in ['time', 'batch']:
            if coord in data_arr.coords:
                data_arr = data_arr.isel({coord: 0})
        # Remove dims of size 1
        data_arr = data_arr.squeeze()
        # Binarize the data
        binarized = xr.where(
                data_arr.isnull(), 
                0., 
                1.,
                ).data 
        # Generate the mask
        mask = gaussian_filter(binarized, sigma=sigma)
        # Convert mask to float32
        mask = mask.astype(np.float32)

        return mask

def replace_norm_drops_vars(norm_matrices: xr.Dataset,
                            new_value: xr.DataArray,
                            vars_to_replace: xr.Dataset,
                            new_var_name: str
                            ) -> xr.Dataset:
    norm_matrices = replace.replace_norm_matrices(
        norm_matrices.rename_vars({"2m_temperature": 'analysed_sst'}),
        new_value,
        vars_to_replace
        )
    normvar2drop = (
        set(norm_matrices.variables.keys()) 
        - set([new_var_name,
               'level',
               'land_sea_mask',
               'year_progress', 
               'year_progress_sin',
               'year_progress_cos', 
               'day_progress',
               'day_progress_sin', 
               'day_progress_cos'])
        )
    norm_matrices = norm_matrices.drop_vars(list(normvar2drop))
    norm_matrices = norm_matrices.drop_dims('level')

    return norm_matrices

class DataSplitter:
    """Class to split the data into train, validation and test sets.
    
    Attributes:
    - dataset (xr.Dataset): The dataset to split.
    - train_size (float): The proportion of the dataset to include in the train split.
    - validation_size (float): The proportion of the dataset to include in the validation split.
    - sampleset_size (int): The size of the sample set.

    Methods:
    - split_data: Slice the dataset by dates and split it into batches.
    - get_split_years: Get the split years for the dataset.
    - split_in_batch: Split the dataset into batches.
    - slice_by_date: Slice the dataset by dates.
    """

    def __init__(
        self, 
        dataset: xr.Dataset, 
        train_size: float = 0.8, 
        validation_size: float = 0.1,
        ) -> None:
        """Constructor of the class.

        Args:
        - dataset (xr.Dataset): The dataset to split.
        - train_size (float): The proportion of the dataset to include in the train split.
        - validation_size (float): The proportion of the dataset to include in the validation split.
        - sampleset_size (int): The size of the sample set.
        """

        self.dataset = dataset
        self.train_size = train_size
        self.validation_size = validation_size

    def split_data(self) -> xr.Dataset:
        """Slice the dataset by dates and split it into batches.

        Args:
        - dates (np.ndarray[str]): The dates to slice the dataset.

        Returns:
        - xr.Dataset: The dataset split into batches.
        """
        # Get the split years
        train_dates, validation_dates, test_dates = self.get_split_years()
        # Slice the dataset by dates
        train_example_batch = self.slice_by_date(train_dates)
        val_example_batch = self.slice_by_date(validation_dates)
        test_example_batch = self.slice_by_date(test_dates)

        return train_example_batch, val_example_batch, test_example_batch

    def get_split_years(self) -> tuple[np.ndarray[str]]:
        """Get the split years for the dataset.

        Returns:
        - tuple[np.ndarray[str]]: The split years for the dataset.
        """

        dataset = self.dataset
        train_size = self.train_size
        validation_size = self.validation_size

        # Get the years from the time dimension
        time_years = dataset.datetime.dt.year
        # Get the unique years
        unique_years = np.unique(time_years.data)
        # Get train set: the 80% of the data
        end_years_train = int(0.8*len(unique_years))
        train_years = unique_years[:end_years_train]
        # Get validation set: the 10% of the data
        end_validation_years = int(0.9*len(unique_years))
        validation_years = unique_years[end_years_train:end_validation_years]
        # Get test set: the 10% of the data
        test_years = unique_years[end_validation_years:]

        return train_years, validation_years, test_years

    def slice_by_date(
        self,
        dates: np.ndarray[str]
        ) -> xr.Dataset:
        """Slice the dataset by dates.

        Args:
        - dates (np.ndarray[str]): The dates to slice the dataset.

        Returns:
        - xr.Dataset: The dataset sliced by dates.
        """

        example_batch: xr.Dataset = self.fill_nan() 
        # Get the first and last date
        firs_year, last_year = np.min(dates), np.max(dates)
        # Get the index of the first and last date
        first_year_idx = np.argwhere(
            example_batch['datetime'].dt.year.data == firs_year
            )[:,1][0]
        last_year_idx = np.argwhere(
            example_batch['datetime'].dt.year.data == last_year
            )[:,1][-1]
        
        return example_batch.isel(time=slice(first_year_idx, last_year_idx))
    
    # Fill nan with min value
    def fill_nan(self, fill_value: str = "mean") -> xr.Dataset:
        dataset = self.dataset
        if fill_value == "mean":
            fill_value_by_var = {
                var: self.dataset[var].mean() 
                for var in self.dataset.variables 
                if self.dataset[var].isnull().any()
                }
        if fill_value == "min":
            fill_value_by_var = {
                var: self.dataset[var].min() 
                for var in self.dataset.variables 
                if self.dataset[var].isnull().any()
                }
        if fill_value != "mean" and fill_value != "min":
            # rise a nonimplemented error
            raise NotImplementedError(f"Fill value {fill_value} is not implemented")
            
        for key, value in fill_value_by_var.items():
            dataset[key] = dataset[key].fillna(value)

        return dataset


class BatchGenerator:
    """Class to generate batches from a dataset.

    Attributes:
    - example_batch (xr.Dataset): The example batch to generate the batches.
    - min_batch_size (int): The minimum batch size.
    - gentype (str): The type of generator.
    - shuffle (bool): Whether to shuffle the batches.

    Methods:
    - generate: Generate the batches.
    - get_minibatch_from_list: Get the minibatch from a list.
    """
    def __init__(
        self, 
        example_batch: xr.Dataset, 
        min_batch_size: int = 2, 
        sliding_window: int = 2,
        gentype: str = 'cyclic', 
        shuffle: bool = True
        ) -> None:
        """Constructor of the class.

        Args:
        - example_batch (xr.Dataset): The example batch to generate the batches.
        - min_batch_size (int): The minimum batch size.
        - gentype (str): The type of generator.
        - shuffle (bool): Whether to shuffle the batches.
        """
        self.example_batch = example_batch
        self.min_batch_size = min_batch_size
        self.gentype = gentype
        self.shuffle = shuffle
        self.sliding_window = sliding_window

        # self.batch_size_from_gen = 0

    def generate(self) -> itertools.cycle or list_iterator:
        """Generate the batches.

        Returns:
        - itertools.cycle or list_iterator: The iterator of the batches.
        """
        # data_batched = self.split_in_batch(self.example_batch)
        batch_list = self.split_in_batch(self.example_batch)
        # Get the number of batches
        # n_current_batch = data_batched.dims['batch']
        # n_current_batch = data_batched.dims['batch']
        # Get the list of batches
        # batch_list = [
        #     data_batched.isel(batch=slice(i, i+1)) 
        #     for i in range(0, n_current_batch)
        #     ]
        # Remove data_batched to free memory
        #del data_batched
        # Shuffle the list of batches
        # np.random.seed(0)  # For reproducibility
        if self.shuffle:
            np.random.shuffle(batch_list)  # Modify inplace the list of batch
        # Get the minibatch from the list
        minibatch_list = self.get_minibatch_from_list(
            batch_list, 
            self.min_batch_size
            )
        # Concatenate the minibatches
        list_batch_minibatch = [
            xr.concat([*b], dim='batch') 
            for b in minibatch_list
            ]
        # Get the batch size from the generator
        self.batch_size_from_gen = len(list_batch_minibatch)
        # Return the iterator
        match self.gentype:
            case 'cyclic':

                return itertools.cycle(list_batch_minibatch)  # Cyclic iterator
            case 'linear':

                return iter(list_batch_minibatch)  # Linear iterator
            case _:

                raise ValueError(f"Invalid generator type: {self.gentype}")
    
    def split_in_batch(self, sliced_dataset: xr.Dataset) -> xr.Dataset:
        """Split the dataset into batches.

        Args:
        - sliced_dataset (xr.Dataset): The dataset to split.

        Returns:
        - xr.Dataset: The dataset split into batches.
        """

        window_size = self.sliding_window
        WINDOW_STEP = 1 
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
            min_reset = sliced_dataset.isel(time=slice(window[0],window[1])).time.min()
            other_batch = sliced_dataset.isel(time=slice(window[0],window[1]))
            other_batch = other_batch.assign_coords(
                time=(other_batch.time - min_reset)
                )
            batch_list.append(other_batch)

        return batch_list #xr.concat(batch_list, dim='batch')

    def get_minibatch_from_list(
        self, 
        batch_list: list, 
        mini_batch_size: int
        ) -> list[list[xr.Dataset]]:
        """Get the minibatch from a list.

        Args:
        - batch_list (list): The list of batches.
        - mini_batch_size (int): The mini batch size.

        Returns:
        - list[list[xr.Dataset]]: The minibatch from the list.
        """
        # Get the last index
        last_i = (len(batch_list) // mini_batch_size) * mini_batch_size
        if last_i < len(batch_list):
            residual_size = len(batch_list) % mini_batch_size
            n_paddding = mini_batch_size - residual_size
            # Padding the last batch repeating the last element
            residual_batch = [
                tuple(batch_list[last_i:] + [batch_list[-1]] * n_paddding)
                ]

        # Get the pairs
        pairs = (
            [tuple(batch_list[i:i+mini_batch_size]) 
            for i in range(0, last_i, mini_batch_size)] 
            + (residual_batch if last_i < len(batch_list) else [])
            )

        return pairs

def device_generator(batch_generator: iter,
                     n_devices: int) -> iter:
    """
    Groups mini-batches into sets to be distributed across devices.

    Args:
    - batch_generator (iter): An iterator producing mini-batches.
    - n_devices (int): The number of devices available.

    Returns:
    - iter: A new iterator producing grouped mini-batches.
    """

    def batch_concatenator():
        batch_list = []
        
        for batch in batch_generator:
            batch_list.append(batch)
            
            # If we have collected enough mini-batches for all devices
            if len(batch_list) == n_devices:
                # Concatenate the mini-batches along a new axis called 'devices'
                concat_batch = xr.concat(batch_list, dim='devices')
                
                # Reset the list for the next group of mini-batches
                batch_list = []
                
                yield concat_batch

        # In case there are remaining mini-batches after the last iteration
        if batch_list:
            # Concatenate the remaining mini-batches
            concat_batch = xr.concat(batch_list, dim='device')
            yield concat_batch
    
    return batch_concatenator()


# Deprecated

"""def calculate_split_dates(datetime_array: xr.DataArray, train_size: float = 0.8, validation_size: float = 0.1) -> tuple[str]:
    total_size = datetime_array.size
    
    # Cálculo de los índices de los conjuntos de datos
    train_end_idx = int(np.round(total_size * train_size))
    validation_end_idx = train_end_idx + int(np.round(total_size * validation_size))
    
    # Obtención de fechas para cada conjunto de datos
    train_dates = (str(datetime_array.isel(time=0).data[0]), 
                   str(datetime_array.isel(time=train_end_idx - 1).data[0]))

    validation_dates = (str(datetime_array.isel(time=train_end_idx).data[0]), 
                        str(datetime_array.isel(time=validation_end_idx - 1).data[0]))

    test_dates = (str(datetime_array.isel(time=validation_end_idx).data[0]), 
                  str(datetime_array.isel(time=-1).data[0]))

    return train_dates, validation_dates, test_dates"""