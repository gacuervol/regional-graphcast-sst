import os
import jax
import numpy as np
import jax.numpy as jnp
import xarray as xr
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
from flax.training import train_state
from typing import Optional, Tuple, List
from dataclasses import dataclass

class ModelRestorer:
    """Class to restore a model from a checkpoint"""
    
    def __init__(self, base_path, params_folder):
        self.path = os.path.join(base_path, params_folder)
        self.checkpointer = ocp.StandardCheckpointer()
       
    def restore(self, input_shape=(8, 2, 256, 256, 1), learning_rate=1e-2):
        """Restore a model from a checkpoint
        Args:
            - input_shape: tuple, shape of the input data (b, t, h, w, c)
            - learning_rate: float, learning rate for the optimizer
        Returns:
            - train_state: TrainState object with the restored model
        """
        # Restore raw parameters
        raw_restored = self.checkpointer.restore(self.path)
       
        # Initialize model
        carry_rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 2)
        model = ConvLSTMBlock(carry_rng, input_shape)
        
        # Configure optimizer
        optimizer = optax.adamw(
            learning_rate,
            b1=0.9,
            b2=0.95,
            eps=1e-08,
            eps_root=0.0,
            mu_dtype=None, 
            weight_decay=0.1
            )
        
        # Create training state
        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=raw_restored['params'],
            tx=optimizer
            )


class Normalizer:
    """Class to normalize and unnormalize variables"""

    def __init__(
        self, 
        scales: jax.Array, 
        locations: jax.Array,
        residual_scales: jax.Array,
        residual_locations: Optional[jax.Array] = None,
        ) -> None:

        self.scales = scales
        self.locations = locations
        self.residual_scales = residual_scales
        self.residual_locations = residual_locations

    def normalize(
        self,
        values: jax.Array,
        ) -> jax.Array:
        """Normalize variables using the given scales and (optionally) locations."""

        norm_array = ( 
        (values - self.locations.astype(values.dtype)) 
        / self.scales.astype(values.dtype)
        )

        return norm_array
    def unnormalize(
        self,
        values: jax.Array,
        ) -> jax.Array:
        """Unnormalize variables using the given scales and (optionally) locations."""

        unorm_values = (
            (values * self.scales.astype(values.dtype))
            + self.locations.astype(values.dtype)
            )
        
        return unorm_values

    def normalize_residual(
        self,
        values: jax.Array,
        residual_locations: Optional[jax.Array] = None,
        ) -> jax.Array:

        if self.residual_locations is not None:
            values = (
                values - self.residual_locations.astype(values.dtype)
                )

        norm_array = (
            values / self.residual_scales.astype(values.dtype)
            )

        return norm_array

    def unnormalize_residual(
        self,
        values: jax.Array,
        residual_locations: Optional[jax.Array] = None,
        ) -> jax.Array:
        """Unnormalize variables using the given scales and (optionally) locations."""

        unorm_values = (
            values * self.residual_scales.astype(values.dtype)
            )

        if self.residual_locations is not None:
            unorm_values = (
                unorm_values + self.residual_locations.astype(values.dtype)
                )

        return unorm_values

@dataclass
class DatasetDimensionsKeeper:
    """Class for store dataset dimension and coordinates"""
    dims: dict
    coords: dict

    def __post_init__(self):
        self.INPUT_DAYS = 2
        self.WINDOW_STEP = 1
        self.dims['time'] -= self.INPUT_DAYS
        self.coords['time'] = self.coords['time'][2:]
        self.coords['datetime'] = self.coords['datetime'][:,self.INPUT_DAYS:]

    def compute_datetime(self, n_times_batch: int) -> np.ndarray:
        """Compute the datetime array"""
        b = self.dims['batch'] 
        datetime = self.coords['datetime']
        datetime_accum = []
        for _ in range(int(n_times_batch)):
            datetime_accum.append(datetime)
            datetime = datetime + np.timedelta64(b * self.WINDOW_STEP, 'D')
        return np.concatenate(datetime_accum, axis=0)


def pipeline_convlstm(
    test_dataset_batched: List[xr.Dataset],
    norms_factors: dict,
    autoregres_steps: int,
    state_restored: train_state,
    device: jax.Device,
    ) -> xr.Dataset:
    """Pipeline to perform a ConvLSTM prediction"""
    # Create the normalizer
    normalizer = Normalizer(**norms_factors)
    dims_and_coords = DatasetDimensionsKeeper(
        dict(test_dataset_batched[0].dims),
        dict(test_dataset_batched[0].coords)
        )
    # Perform the prediction per batch
    all_predictions = []
    for batch in test_dataset_batched:
        t_steps = 1
        reshape_input = (8, 2, 256, 256, 1)
        inputs_norm, targets_norm, _ = normalize_and_reshape_batch(
                        batch, 
                        normalizer, 
                        t_steps, 
                        reshape_input,
                        )
        predictions_10D = rollout(
            state_restored,
            inputs_norm,
            autoregres_steps,
            normalizer,
            device,
            )
        # Append the predictions
        all_predictions.append(
            jnp.stack(predictions_10D, axis=1).squeeze()
            )
    # Convert the predictions to xarray    
    convlstm_preds = jax_pred_to_xr(
        all_predictions, 
        dims_and_coords,
        )

    return convlstm_preds


def rollout(
    model_state: train_state.TrainState,
    inputs: jax.Array,
    steps: int,
    normalizer: Normalizer,
    device: jax.Device,
    ) -> List:
    """Wrapper for the model function to perform a rollout.

    Args:
        model_state: The model to use for the rollout.
        inputs: The initial inputs to the model.
        steps: The number of steps to rollout.
        normalizer: The normalizer to use for the model.

    Returns:
        A list of outputs from the model.  
    """
     # Perform the rollout
    outputs = []
    for _ in range(steps):
        predictions_unnorm = inference(
            model_state, 
            inputs, 
            normalizer,
            device,
            )
        outputs.append(jnp.expand_dims(predictions_unnorm, axis=(1,)))
        predictions_norm = normalizer.normalize(predictions_unnorm)
        inputs = jnp.concatenate(
            [
                inputs.at[:, 1:, :, :, :].get(), 
                jnp.expand_dims(predictions_norm, axis=(1, -1))
            ], 
            axis=1
            )

    return outputs


def jax_pred_to_xr(
    all_predictions: list, 
    dims_and_coords: DatasetDimensionsKeeper,
    ) -> xr.Dataset:
    """Convert the predictions to a xarray Dataset"""
    # Concatenate the predictions
    # all_predictions = jnp.concatenate(all_predictions, axis=0)
    # Resize the predictions
    b, t, _, _ = all_predictions[0].shape
    _, _, h, w = dims_and_coords.dims.values() # batch, time, lat, lon
    # Resize the predictions
    all_predictions_resized = []
    for pred in all_predictions:
        # Resize the predictions
        pred = jax.image.resize(pred, (b, t, h, w), "tricubic")
        # Convert to numpy
        pred = np.array(pred)
        all_predictions_resized.append(pred)
    del all_predictions
    # Concatenate the predictions
    n_times_batch = len(all_predictions_resized)
    all_predictions_resized = np.concatenate(all_predictions_resized, axis=0)
    # Convert to xarray
    all_predictions_resized = xr.Dataset(
        {
            'analysed_sst': (list(dims_and_coords.dims.keys()), all_predictions_resized)
        },
        coords = {
        'lat': dims_and_coords.coords['lat'],
        'lon': dims_and_coords.coords['lon'],
        'time': dims_and_coords.coords['time'],
        'datetime': (['batch', 'time'], dims_and_coords.compute_datetime(n_times_batch))
        }
    )
    
    return all_predictions_resized


def inference(
    state: train_state.TrainState, 
    inputs_norm: jax.Array, 
    normalizer: Normalizer,
    device: jax.Device, 
    )-> tuple[train_state.TrainState, jax.Array, jax.Array]:
    """Train for a single step."""
    # Predict the delta x
    params_in_device = jax.device_put(state.params, device)
    inputs_norm_in_device = jax.device_put(inputs_norm, device)
    dx_predictions_norm = state.apply_fn(params_in_device, inputs_norm_in_device)
    dx_predictions = normalizer.unnormalize_residual(dx_predictions_norm).squeeze()
    # unnormalize the the input data
    inputs = normalizer.unnormalize(inputs_norm)
    last_input = inputs.at[:, -1, :, :, :].get().squeeze()
    predictions = last_input + dx_predictions

    return predictions


class ConvLSTMBlock(nn.Module):
    """Convolutional LSTM block."""
    carry_rng: jax.random.PRNGKey
    input_shape: Tuple[int, int, int, int, int]
    
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


def normalize_and_reshape_batch(batch, normalizer, t_steps, reshape_input):
    """Normalize and reshape the input data

    Args:
        - batch: xarray dataset with the input data
        - normalizer: Normalizer object
        - t_steps: int, number of time steps to predict
        - reshape_input: tuple, shape of the input data (b, t, h, w, c)
    Returns:
        - x_jax_resized: jax array, normalized and reshaped input data
        - y_jax_resized: jax array, normalized and reshaped target data
    """

    # Extract the input and target data
    inputs = batch['analysed_sst'].isel(time=slice(None, 2))
    targets = batch['analysed_sst'].isel(time=slice(2, 2 + t_steps))
    # Uncast the input data and convert to JAX array
    inputs_caster, targets_caster = (
        XarrayCaster(inputs),
        XarrayCaster(targets),
    )
    inputs, targets = (
        inputs_caster.uncast_xarray(),
        targets_caster.uncast_xarray(),
    )
    # Convert to JAX array
    inputs = jnp.array(inputs)
    targets = jnp.array(targets)
    dx_target = targets - inputs.at[:, 1:2, :, :].get()
    # Normalize the input data
    inputs_norm = normalizer.normalize(inputs)
    targets_norm = normalizer.normalize(targets)
    dx_target_norm = normalizer.normalize_residual(dx_target)
    # Reshape the input data
    b, t, h, w, c = reshape_input
    inputs_norm_resized = JaxArrayResizer(inputs_norm).reshape_input((h, w))
    targets_norm_resized = JaxArrayResizer(targets_norm).reshape_input((h, w))
    dx_target_norm_resized = JaxArrayResizer(dx_target_norm).reshape_input((h, w))

    return inputs_norm_resized, targets_norm_resized, dx_target_norm_resized


class XarrayCaster:
    """Class to cast an xarray DataArray to a numpy array and viceversa"""

    def __init__(self, xarr: xr.DataArray):
        self.xarr = xarr
    
    def uncast_xarray(self) -> tuple[xr.DataArray, np.ndarray]:

        cast_xarray = self.xarr.isel(time=slice(1, 2))
        cast_xarray['datetime'].values = cast_xarray.datetime.values + np.timedelta64(1, 'D') 
        self.cast_xarray = cast_xarray * 0

        return self.xarr.data

    def cast_xarray(self, x: np.ndarray) -> xr.DataArray:
        self.cast_xarray.values = x

        return self.cast_xarray


class JaxArrayResizer:
    """Class to resize a jax array"""

    def __init__(self, arr: jnp.array):
        self.arr = arr
    
    # Reshape the input data to match the expected shape
    def reshape_input(
        self,
        target_latlon_shape: tuple[int, int],
        ) -> jnp.array:

        x = self.arr
        # target_latlon_shape = (500, 600)
        assert len(target_latlon_shape) == 2, "target_latlon_shape must be a tuple of two integers"
        new_h = target_latlon_shape[0]
        new_w = target_latlon_shape[1]
        # Add channel dimension
        x_jax_expanded = jnp.expand_dims(x, axis=-1)
        b, t, h, w, c = x_jax_expanded.shape
        x_jax_resized = jax.image.resize(
            x_jax_expanded, 
            (b, t, new_h, new_w, c),
            "tricubic",
            )
        return x_jax_resized
    
    def resize_to_power_2(
        self,
        target_size: str | tuple[int, int] = 'power_of_2',
        ) -> jnp.array:

        x = self.arr
        b, t, h, w, c = x.shape
        
        if target_size == 'power_of_2':
            new_shape = self.get_nearest_power2_shape(x)
        elif isinstance(target_size, tuple) and len(target_size) == 2 and all(isinstance(dim, int) for dim in target_size):
            new_shape = (b, t, target_size[0], target_size[1], c)
        else:
            warnings.warn("target_size must be 'power_of_2' or a tuple of two even integers", UserWarning)
            # Use default behavior or return None, depending on your needs
        
        resized = jnp.zeros(new_shape, dtype=x.dtype)
        
        for i, j, k in itertools.product(range(b), range(t), range(c)):
            slice_2d = x[i, j, :, :, k]
            # x_arr = np.expand_dims(np.array(slice_2d), axis=-1)
            x_arr = jnp.expand_dims(slice_2d, axis=-1)
            # x_torh = torch.from_numpy(x_arr)
            # h, w, c = x_torh.shape
            h, w, c = x_arr.shape
            x_arr = jnp.reshape(x_arr, (c, h, w))
            resized_slice = jnp.image.resize(
                x_arr, 
                (new_shape[2], new_shape[3]),
                "tricubic",
                )
            # resized[i, j, :, :, k] = resized_slice
            resized.at[i, j, :, :, k].set(resized_slice)
        
        return resized

    def get_nearest_power2_shape(self, x: jnp.array) -> tuple:
        
        b, t, h, w, c = x.shape
        new_shape = lambda n: 2 ** int(np.round(np.log2(n)))

        return (b, t, new_shape(h), new_shape(w), c)