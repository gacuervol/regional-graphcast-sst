import pytest
import numpy as np
import sys, os
# Obtener el nombre del usuario actual
if sys.platform.startswith("linux"):
    usuario_actual = os.path.expanduser("~").split('/')[-1]
elif sys.platform.startswith("win"):
    usuario_actual = os.path.expanduser("~").split('\\')[-1]
else:
    raise EnvironmentError("Unsupported OS")

sys.path.append(f"/home/{usuario_actual}/PhD_repo/tests") 
from tests_services import fake_dataset

sys.path.append(f"/home/{usuario_actual}/PhD_repo/src") 
from art1_tools.plot_utils import BarrierPlotDataExtractor

def test_BarrierPlotDataExtractor():
    # Define the fake model RMSE
    model_rmse = fake_dataset(shape=(70, 20, 300, 300))['analysed_sst'].isel(lat=0, lon=0)
    # Initialize the extractor
    n_lead_times = model_rmse.sizes['time']
    extractor = BarrierPlotDataExtractor(model_rmse, n_lead_times)
    # Final time coord size
    lead_time_size = extractor.LEAD_TIMES // extractor.WINDOW_STEP
    # Compute the barrier data
    barrier_data = extractor()
    # Test the shape and the dates
    # -----------------------------
    min_date = model_rmse.datetime.min().data.astype('datetime64[D]')
    max_date = model_rmse.datetime.max().data.astype('datetime64[D]')
    dates_coord = np.arange(min_date, max_date)
    assert np.all(barrier_data.date == dates_coord), \
        f"The dates are not the same: expected {np.arange(min_date, max_date)} but got {barrier_data.date}"
    assert barrier_data.shape == (len(dates_coord), lead_time_size), \
        f"Expected shape ({len(dates_coord), lead_time_size}) but got {barrier_data.shape}"