import sys, os
import pytest
import numpy as np
import jax.numpy as jnp
import xarray as xr
import jax
# Obtener el nombre del usuario actual
if sys.platform.startswith("linux"):
    usuario_actual = os.path.expanduser("~").split('/')[-1]
elif sys.platform.startswith("win"):
    usuario_actual = os.path.expanduser("~").split('\\')[-1]
else:
    raise EnvironmentError("Unsupported OS")

sys.path.append(f"/home/{usuario_actual}/PhD_repo/src") 
from art1_tools.legacy_model import DatasetDimensionsKeeper, pipeline_convlstm

sys.path.append(f"/home/{usuario_actual}/PhD_repo/tests") 
from tests_services import fake_dataset, fake_convlstm, fake_norms, fake_test_generator

def test_DatasetDimensionsKeeper():
    fake_baseline = fake_dataset(
        shape=(724, 12, 300, 300), 
        start_date=np.datetime64('2017-01-01')
        )
    fake_baseline.assign_coords(
        time=(['time'], fake_baseline.time.data + np.timedelta64(0, 'D'))
        )
    dims_and_coords = DatasetDimensionsKeeper(
        dict(fake_baseline.isel(batch=slice(None, 8)).dims),
        dict(fake_baseline.isel(batch=slice(None, 8)).coords)
        )
    expected_dims = fake_baseline.isel(batch=slice(None, 8), time=slice(2, None)).dims
    assert dims_and_coords.dims == expected_dims, \
        f"Expected {expected_dims} but got {dims_and_coords.dims}"
    n_repeated_batches = 4
    batches_div8 = fake_baseline.dims['batch'] // 8
    residual = 1 if fake_baseline.dims['batch'] % 8 > 0 else 0
    n_times_batch = batches_div8 + residual
    date_time = dims_and_coords.compute_datetime(n_times_batch)[:-n_repeated_batches]
    assert date_time.shape == fake_baseline['datetime'][:, 2:].shape, \
        f"Expected {fake_baseline['datetime'][:, 2:].shape} but got {date_time.shape}"
    assert np.all(date_time == fake_baseline['datetime'][:, 2:]), \
        "The datetime arrays are not equal"


def test_edge2edge_legacy_fns():
    # Restore the legacy model
    fake_baseline = fake_dataset(
        shape=(80, 12, 300, 300), 
        start_date=np.datetime64('2017-01-01')
        )
    state_restored = fake_convlstm()
    stddev_by_level, mean_by_level, diffs_stddev_by_level = fake_norms()
    # Define normalization factors
    norms_factors = {
        'scales': jnp.array(stddev_by_level['analysed_sst'].data),
        'locations': jnp.array(mean_by_level['analysed_sst'].data),
        'residual_scales': jnp.array(diffs_stddev_by_level['analysed_sst'].data),
        }
    fake_test_gen = fake_test_generator()
    n_batches = (
        (fake_baseline.dims['batch'] // 8)
        + (1 if fake_baseline.dims['batch'] % 8 > 0 else 0)
        )
    fake_test_gen = [next(fake_test_gen) for _ in range(n_batches)]
    autoregres_steps = 10
    convlstm_preds = pipeline_convlstm(
        fake_test_gen,
        norms_factors,
        autoregres_steps,
        state_restored,
        jax.devices("cpu")[0],
        )
    assert isinstance(convlstm_preds, xr.Dataset), \
        f"Expected xr.Dataset but got {type(convlstm_preds)}"
    assert list(convlstm_preds['analysed_sst'].dims) == list(fake_baseline.dims), \
        f"Expected {list(fake_baseline.dims)} but got {list(convlstm_preds['analysed_sst'].dims)}"
    assert np.all(convlstm_preds['analysed_sst'].datetime.data == fake_baseline['datetime'][:, 2:].data), \
        "The datetime arrays are not equal in the predictions"