def slice_by_date(example_batch: xr.Dataset, dates_tuple: tuple[str]) -> xr.Dataset:
    start_date, stop_date = dates_tuple 
    id_star = (example_batch['datetime'].data == pd.to_datetime(start_date)).argmax()
    id_stop = (example_batch['datetime'].data == pd.to_datetime(stop_date)).argmax()
    return example_batch.isel(time=slice(id_star, id_stop + 1))

def split_in_batch(dataset: xr.Dataset, n_sample: int = 2) -> xr.Dataset:
    input_steps = 2
    window_size = n_sample
    # window_size = dataset.time.size - (input_steps * sample_size)
    batch_size = (dataset.time.size - window_size) / input_steps
    batch_list = []
    for batch in np.arange(batch_size):
        step_size = int(input_steps * batch if batch > 0 else 0)
        window = np.array([0, window_size]) + step_size
        min_reset = dataset.isel(time=slice(window[0],window[1])).time.min()
        other_batch = dataset.isel(time=slice(window[0],window[1]))
        other_batch = other_batch.assign_coords(time=(other_batch.time - min_reset))
        batch_list.append(other_batch)
    return xr.concat(batch_list, dim='batch')

def calculate_split_dates(datetime_array: xr.DataArray, train_size: float = 0.8, validation_size: float = 0.1) -> tuple[str]:
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

    return train_dates, validation_dates, test_dates

def batch_generator(example_batch: xr.Dataset, 
                    min_batch_size=2, 
                    gentype='cyclic'
                    ) -> itertools.cycle or list_iterator:

    def get_minibatch_from_list(batch_list: list, mini_batch_size: int) -> list[list[xr.Dataset]]: 
        
        last_i = (len(batch_list) // mini_batch_size) * mini_batch_size
        pairs = ([tuple(batch_list[i:i+mini_batch_size]) for i in range(0, last_i, mini_batch_size)] 
                 + ([tuple(batch_list[last_i:])] if last_i < len(batch_list) else []))
        
        return pairs

    n_current_batch = example_batch.dims['batch']
    batch_list = [example_batch.isel(batch=slice(i, i+1)) for i in range(0, n_current_batch)]
    np.random.shuffle(batch_list) # Modify inplace the list of batch
    minibatch_list = get_minibatch_from_list(batch_list, min_batch_size)
    list_batch_minibatch = [xr.concat([*b], dim='batch') for b in minibatch_list]
    global batch_size_from_gen 
    batch_size_from_gen = len(list_batch_minibatch)
    match gentype:
        case 'cyclic':
            return itertools.cycle(list_batch_minibatch)  # Cyclic iterator
        case 'linear':
            return iter(list_batch_minibatch)  # Linear iterator
        case _:
            raise ValueError(f"Invalid generator type: {gentype}")

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