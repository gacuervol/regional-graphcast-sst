import jax
import numpy as np
import xarray as xr
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple
import torch
import itertools
import warnings


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


# def cast_xarray(cast: xr.DataArray, x: np.ndarray) -> xr.DataArray:
#     cast.values = x
# 
#     return cast
# 
# def uncast_xarray(x: xr.DataArray) -> tuple[xr.DataArray, np.ndarray]:
#     cast_xarray = x.isel(time=slice(1, 2))
#     #cast_xarray['time'] = cast_xarray.time + np.timedelta64(1, 'D')
#     cast_xarray['datetime'].values = cast_xarray.datetime.values + np.timedelta64(1, 'D') 
#     cast_xarray = cast_xarray * 0
#     return cast_xarray, x.data

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
