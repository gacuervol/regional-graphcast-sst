import os, sys
import pytest
import xarray as xr
import numpy as np
# Obtener el nombre del usuario actual
if sys.platform.startswith("linux"):
    usuario_actual = os.path.expanduser("~").split('/')[-1]
elif sys.platform.startswith("win"):
    usuario_actual = os.path.expanduser("~").split('\\')[-1]
else:
    raise EnvironmentError("Unsupported OS")

sys.path.append(f"/home/{usuario_actual}/PhD_repo/src") 
sys.path.append(f"/home/{usuario_actual}/PhD_repo/tests") 

from tests_services import fake_dataset, fake_dataset_with_nan
from art1_tools.compatibility import PredictionCompatibilityChecker


def test_PredictionCompatibilityChecker_with_ForecastEvaluator():
    # Test the compatibility checker with the ForecastEvaluator
    # ---------------------------------------------------------
    # 1. run the compatibility checker
    # -----------------------------------
    baseline = fake_dataset()
    models = [
        {
            'name': 'model1',
            'predictions': fake_dataset()
        },
        {
            'name': 'model2',
            'predictions': fake_dataset()
        }]

    checker = PredictionCompatibilityChecker(baseline, models)
    compatible_data = checker.check_compatibility()
    assert bool(compatible_data)
    # 2. run the ForecastEvaluator
    # ---------------------------------
    # Get the current user name
    import os, sys
    # Obtener el nombre del usuario actual
    if sys.platform.startswith("linux"):
        usuario_actual = os.path.expanduser("~").split('/')[-1]
    elif sys.platform.startswith("win"):
        usuario_actual = os.path.expanduser("~").split('\\')[-1]
    else:
        raise EnvironmentError("Unsupported OS")

    sys.path.append(f"/home/{usuario_actual}/PhD_repo/src") 
    from art1_tools import metrics
    var_name = 'analysed_sst'

    for compatible in compatible_data:
        forecast = compatible.forecast_and_truth[0][var_name] 
        truth = compatible.forecast_and_truth[1][var_name] 
        land_mask = fake_dataset_with_nan()[var_name]
        evaluator = metrics.ForecastEvaluator(forecast, truth, land_mask)
        assert isinstance(evaluator.RMSE(), xr.DataArray)