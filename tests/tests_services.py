import jax
import optax
import jax.numpy as jnp
import xarray as xr
import numpy as np
from typing import Tuple
from flax import linen as nn
from flax.training import train_state

def fake_dataset(
    shape: tuple = (728, 10, 300, 300), 
    start_lat: float = 19.57, 
    start_lon: float = -20.93, 
    start_date: np.datetime64 = np.datetime64('2020-01-01'),
    ) -> xr.Dataset:
    """Create a xarray Dataset filled with zeros"""

    b, t, h, w = shape
    data_vars = {
        'analysed_sst': (['batch', 'time', 'lat', 'lon'], np.zeros(shape)),
        'analysis_error': (['batch', 'time', 'lat', 'lon'], np.zeros(shape)),
        'land_sea_mask': (['batch', 'lat', 'lon'], np.ones((b, h, w)))
        }
    # Adjust datetime 
    WINDOW_STEP = 1
    delta_adjustment = np.arange(b) * np.timedelta64(t - WINDOW_STEP, 'D')
    dates = np.arange(
        start_date, start_date + np.timedelta64((b*t), 'D')
        ).reshape(b, t).astype('datetime64[ns]')
    coords = {
        'lat': (["lat"], np.arange(start_lat, start_lat + (h * 0.05), 0.05)),
        'lon': (["lon"], np.arange(start_lon, start_lon + (w * 0.05), 0.05)),
        'time': (["time"], np.arange(np.timedelta64(0, 'D'), np.timedelta64(t, 'D'))),
        'datetime': (["batch", "time"], dates - delta_adjustment[:, np.newaxis].repeat(t, axis=1))
        }

    return xr.Dataset(data_vars=data_vars, coords=coords)

def fake_reanalysis(
    shape: tuple = (180, 180, (728 * 2) + 12 ), 
    start_lat: float = 19.58, 
    start_lon: float = -20.92, 
    start_date: np.datetime64 = np.datetime64('2020-01-01'),
    ) -> xr.Dataset:
    """Create a xarray Dataset filled with zeros"""

    h, w, t = shape
    data_vars = {
        'thetao': (['latitude', 'longitude', 'time'], np.zeros(shape)),
        }
    coords = {
        'latitude': (["latitude"], np.arange(start_lat, start_lat + (h * 0.05), 0.05)),
        'longitude': (["longitude"], np.arange(start_lon, start_lon + (w * 0.05), 0.05)),
        'time': (["time"],  np.arange(start_date, start_date + np.timedelta64((t), 'D')).astype('datetime64[ns]')),
        'depth': (["depth"], np.array([0, 10])),
        }

    ds = xr.Dataset(data_vars=data_vars, coords=coords).isel(depth=0)
    return ds
    
def fake_dataset_with_nan(
    shape: tuple = (300, 300), 
    start_lat: float = 19.57, 
    start_lon: float = -20.93, 
    start_date: np.datetime64 = np.datetime64('2020-01-01'),
    ) -> xr.Dataset:
    """Create a xarray Dataset filled with zeros"""

    h, w = shape
    random_array = np.random.randint(-1, 1, size=shape)
    data_vars = {
        'analysed_sst': (['lat', 'lon'], np.where(random_array < 0, np.nan, random_array)),
        'analysis_error': (['lat', 'lon'], np.where(random_array < 0, np.nan, random_array)),
        }
    coords = {
        'lat': (["lat"], np.arange(start_lat, start_lat + (h * 0.05), 0.05)),
        'lon': (["lon"], np.arange(start_lon, start_lon + (w * 0.05), 0.05)),
        'time': (["time"], np.arange(np.timedelta64(0, 'D'), np.timedelta64(1, 'D'))),
        }

    ds = xr.Dataset(data_vars=data_vars, coords=coords).squeeze()
    return ds


def fake_norms() -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Create fake norms"""
    stddev_data_vars = {
        'analysed_sst': ([], np.array(2.1082823)),
        'land_sea_mask': ([], np.array(0.49274957)),
        'year_progress': ([], np.array(0.28866217)),
        'year_progress_sin': ([], np.array(0.7071226)),
        'year_progress_cos': ([], np.array(0.7070909)),
        'day_progress': ([], np.array(0.01202807)),
        'day_progress_sin': ([], np.array(0.07337836)),
        'day_progress_cos': ([], np.array(0.01772283)),
        'toa_incident_solar_radiation': ([], np.array(586926.44)),
        }
    mean_data_vars = {
        'analysed_sst': ([], np.array(293.68823)),
        'land_sea_mask': ([], np.array(0.5451222)),
        'year_progress': ([], np.array(0.5000029)),
        'year_progress_sin': ([], np.array(1.2870645e-07)),
        'year_progress_cos': ([], np.array(-4.4826567e-05)),
        'day_progress': ([], np.array(0.462638977)),
        'day_progress_sin': ([], np.array(0.23193312)),
        'day_progress_cos': ([], np.array(-0.9697981)),
        'toa_incident_solar_radiation': ([], np.array(3874307.8)),
        }
    diffs_stddev_data_vars = {
        'analysed_sst': ([], np.array(0.03981644)),
        'land_sea_mask': ([], np.array(0.00782354)),
        'year_progress': ([], np.array(0.05140935)),
        'year_progress_sin': ([], np.array(0.01216325)),
        'year_progress_cos': ([], np.array(0.01216487)),
        'day_progress': ([], np.array(0.)),
        'day_progress_sin': ([], np.array(0.)),
        'day_progress_cos': ([], np.array(0.)),
        'toa_incident_solar_radiation': ([], np.array(10504.8125)),
        }
    # Adjust datetime 
    coords = {}

    stddev_by_level = xr.Dataset(data_vars=stddev_data_vars, coords=coords)
    mean_by_level = xr.Dataset(data_vars=mean_data_vars, coords=coords)
    diffs_stddev_by_level = xr.Dataset(data_vars=diffs_stddev_data_vars, coords=coords)
    return (mean_by_level, stddev_by_level, diffs_stddev_by_level)



def fake_convlstm():
    """Fake ConvLSTM model"""
    # Define the fake model
    class FakeConvLSTMBlock(nn.Module):
        """Convolutional LSTM block."""
        carry_rng: jax.random.PRNGKey
        input_shape: tuple[int, int, int, int, int]

        def setup(self):
        
            self.convlstm1 = nn.ConvLSTMCell(features=8, kernel_size=(3,3), strides=1, padding=1)
            self.convlstm2 = nn.ConvLSTMCell(features=1, kernel_size=(3,3), strides=1, padding=1)
            self.carry = self.initialize_carry(self.carry_rng, input_shape=self.input_shape)
            
        def __call__(self, x: jax.Array):

            carry1, carry2 = self.carry
            b, t, h, w, c = x.shape
            for i in range(t):
                x_t = x.at[:, i:i+1, :, :, :].get()
                carry1, h = self.convlstm1(carry1, x_t)
                carry2, h = self.convlstm2(carry2, h)
            return h

        def initialize_carry(self, rng, input_shape):
            b, t, h, w, c = input_shape
            rng_keys = jax.random.split(rng, 2)
            carry1 = self.convlstm1.initialize_carry(
                rng_keys[0],
                input_shape=(b, 1, h, w, c)
                )
            carry2 = self.convlstm2.initialize_carry(
                rng_keys[1],
                input_shape=(b, 1, h, w, c)
                )

            return carry1, carry2 
    
    # Initialize the fake model
    input_shape = (8, 2, 256, 256, 1)
    carry_rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 2)

    model = FakeConvLSTMBlock(carry_rng, input_shape)
    params = model.init(
        init_rng, 
        jnp.ones(input_shape),
        )
    lr = 1e-2
    optimizer = optax.adamw(
        lr, 
        b1=0.9, 
        b2=0.95, 
        eps=1e-08, 
        eps_root=0.0, 
        mu_dtype=None,
        weight_decay=0.1,
        )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        )
    return state

def fake_test_generator(batch_size: int = 8):
    """Fake test generator"""

    fake_baseline = fake_dataset(
        shape=(728, 12, 300, 300), 
        start_date=np.datetime64('2017-01-01')
        )
    total_batches = fake_baseline.sizes['batch']
    
    for start in range(0, total_batches, batch_size):
        end = start + batch_size
        # Slice the dataset along the batch dimension
        batch_subset = fake_baseline.isel(batch=slice(start, end))
        yield batch_subset