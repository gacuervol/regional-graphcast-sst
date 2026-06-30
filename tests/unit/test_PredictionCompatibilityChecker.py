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

from tests_services import fake_dataset, fake_reanalysis
from art1_tools.compatibility import PredictionCompatibilityChecker
from art1_tools.compatibility import CompatibleData


def test_PredictionCompatibilityChecker_split_in_batch_fn():
    """Test for PredictionCompatibilityChecker.split_in_batch"""
    
    # Test for a dataset with the expected shape
    # --------------------------------------
    # Create a dataset with 10 time steps
    reanalysis = fake_reanalysis()
    # Split the dataset into batches
    reanalysis_batch = PredictionCompatibilityChecker.split_in_batch(reanalysis)
    # Check the shape of the dataset
    assert reanalysis_batch.dims == {'batch': 728, 'latitude': 180, 'longitude': 180, 'time': 10}, \
        f"Se esperaba un shape (728, 180, 180, 10) pero se obtuvo {reanalysis_batch.thetao.shape}"
   
    # Test with the same timedates of the baseline
    # --------------------------------------------
    # Create a dataset with 10 time steps
    baseline = fake_dataset()
    assert baseline.datetime.shape == reanalysis_batch.datetime.shape, \
        f"Se esperaba un shape {baseline.datetime.shape} pero se obtuvo {reanalysis_batch.datetime.shape}"
    assert np.all(baseline.datetime.values == reanalysis_batch.datetime.values), \
        f"Se esperaba {baseline.datetime.values} pero se obtuvo {reanalysis_batch.datetime.values}"

def test_PredictionCompatibilityChecker_check_for_reanalysis_data():
    """Test for PredictionCompatibilityChecker.check_compatibility with reanalysis data"""
    
    # Test different shapes
    # --------------------- 
    # Crear un baseline de reanalysis
    baseline = fake_dataset()
    
    # Crear un modelo de reanalysis
    models = [{
            'name': 'reanalysis',
            'predictions': fake_reanalysis(
                shape=(180, 180, 922), 
                start_date=np.datetime64('2022-06-01')
                )
        }]
    # Crear el checker
    checker = PredictionCompatibilityChecker(baseline, models)
    # Verificar compatibilidad
    with pytest.raises(ValueError) as exc_info:

        compatible_data = checker.check_compatibility()
        
    assert exc_info.type is ValueError, f"Se esperaba un ValueError pero se obtuvo {exc_info.type}"
    assert "operands could not be broadcast together with shapes" in str(exc_info.value), \
        f"Se esperaba un: \nValueError: operands could not be broadcast... pero se obtuvo {exc_info.value}"

    # Test same shapes but different dates
    # ------------------------------------
    # Crear un baseline de reanalysis
    baseline = fake_dataset()
    
    # Crear un modelo de reanalysis
    models = [{
            'name': 'reanalysis',
            'predictions': fake_reanalysis(start_date=np.datetime64('2022-06-01'))
        }]
    # Crear el checker
    checker = PredictionCompatibilityChecker(baseline, models)

    # Verificar compatibilidad
    with pytest.raises(ValueError) as exc_info:

        compatible_data = checker.check_compatibility()

    assert exc_info.type is ValueError, f"Se esperaba un ValueError pero se obtuvo {exc_info.type}"
    assert "Las fechas no coinciden para el modelo reanalysis" in str(exc_info.value), \
        f"Se esperaba un: \nValueError: Las fechas no coinciden para el modelo reanalysis... pero se obtuvo {exc_info.value}"

def test_PredictionCompatibilityChecker_check_different_dates():
    # Test different dates
    # ---------------------
    baseline = fake_dataset()
    models = [{
            'name': 'model1',
            'predictions': fake_dataset(start_date=np.datetime64('2020-01-02'))
        }]

    checker = PredictionCompatibilityChecker(baseline, models)
    try:
        checker.check_compatibility()
    except ValueError as e:
        assert type(e) == ValueError
    # Test equal dates
    # ----------------    
    baseline = fake_dataset()
    models = [{
            'name': 'model1',
            'predictions': fake_dataset()
        }]

    checker = PredictionCompatibilityChecker(baseline, models)
    assert bool(checker.check_compatibility())

def test_PredictionCompatibilityChecker_check_expected_length():
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
    assert len(checker.check_compatibility()) == len(models), f"The length of the list is {len(checker.check_compatibility())} and should be {len(models)}"

def test_PredictionCompatibilityChecker_check_expected_return_type():
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
    assert isinstance(checker.check_compatibility(), list), f"The type of the return is {type(checker.check_compatibility())} and should be {list}"

    for i in range(len(checker.check_compatibility())):
        # Check the type of the elements in the list
        assert type(checker.check_compatibility()[i]) == CompatibleData
        # Check the type of the attributes of the elements in the list
        assert type(checker.check_compatibility()[i].name) == str
        print(f":: Model name: {checker.check_compatibility()[i].name}")
        # Check the type of the return of the method initial_date
        assert type(checker.check_compatibility()[i].initial_date()) == np.ndarray
        # Check the type of the elements in the return of the method initial_date
        assert checker.check_compatibility()[i].initial_date().dtype == 'datetime64[ns]'
        # Check the type of the elements in the return of the method check_compatibility
        assert type(checker.check_compatibility()[i].forecast_and_truth) == tuple
        # Check the type of the elements in the return of the method check_compatibility
        assert type(checker.check_compatibility()[i].forecast_and_truth[0]) == xr.Dataset
        # Check the type of the elements in the return of the method check_compatibility
        assert type(checker.check_compatibility()[i].forecast_and_truth[1]) == xr.Dataset