import jax
import numpy as np
import jax.numpy as jnp
import xarray as xr
import haiku as hk
import json
import shutil
import optax
from datetime import datetime
from jaxlib.xla_extension import ArrayImpl
from jaxlib.xla_extension import XlaRuntimeError
import functools
import itertools
import dataclasses
import os, sys
from scipy.ndimage import distance_transform_edt
sys.path.append("/home/user/PhD_repo/src") 
sys.path.append('/home/user/PhD_repo/models')
from art1_tools import graphcast_newvars
from art1_tools import casting_newvars
from art1_tools import data_utils_newvars
from art1_tools import normalization_newvars
from art1_tools import autoregressive_newvars
from art1_tools import xarray_jax_newvars
from art1_tools import data_load
from art1_tools import metrics

from Graphcast_model import xarray_tree
from Graphcast_model import checkpoint
from Graphcast_model import rollout


def construct_wrapped_graphcast(model_config: graphcast_newvars.ModelConfig,
                                task_config: graphcast_newvars.TaskConfig):
    """
    Construye y envuelve el Predictor de GraphCast_graphcast_newvars.
    """
    # Predictor más profundo de un paso.
    predictor = graphcast_newvars.GraphCast(model_config, task_config)

    # Modifica las entradas/salidas a `graphcast.GraphCast` para manejar la conversión de/a
    # float32 de/a BFloat16.
    predictor = casting_newvars.Bfloat16Cast(predictor)

    # Modifica las entradas/salidas a `casting.Bfloat16Cast` para que la conversión de/a
    # BFloat16 ocurra después de aplicar la normalización a las entradas/objetivos.
    predictor = normalization_newvars.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level)

    # Envuelve todo para que el modelo de un paso pueda producir trayectorias.
    predictor = autoregressive_newvars.Predictor(
        predictor, gradient_checkpointing=True)
    return predictor

# Transformación de Haiku que envuelve el predictor para ejecutar hacia adelante.
@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    """Realiza una ejecución hacia adelante del modelo predictor."""
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template, forcings)


# Transformación de Haiku que envuelve la función de pérdida.
@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings, weights):
    """Calcula la pérdida del modelo predictor."""
    predictor = construct_wrapped_graphcast(model_config, task_config)
    # jax.debug.print("targets antes de entrar 🤯 {} 🤯", targets)
    loss, diagnostics = predictor.loss(inputs, targets, forcings, weights)
    jax.debug.print('\t# loss per sample : {}', loss)
    return xarray_tree.map_structure(
        lambda x: xarray_jax_newvars.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics)) # Mean batch


def grads_fn(params, state, model_config, task_config, inputs, targets, forcings, weights):
    """Calcula los gradientes de la función de pérdida."""
    if weights is not None:
        assert (len(weights.dims) == 2), f'weights must have size dim of 2 but it is: {weights.dims}'
    def _aux(params, state, i, t, f, w):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(0), model_config, task_config,
            i, t, f, w)
        return loss, (diagnostics, next_state)
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
        _aux, has_aux=True)(params, state, inputs, targets, forcings, weights)
    return loss, diagnostics, next_state, grads
        
# Compilar la función de pérdida y los gradientes con Haiku, JAX JIT y las configuraciones del modelo y la tarea.
def data_model_config(cycle: int, M_i: int, msg_steps: int, 
                      latent_size: int, train_data: xr.Dataset,
                      ) -> tuple[graphcast_newvars.ModelConfig, None, dict]:
    # Hyperparams block
    source = "Random"
    random_mesh_size = M_i # How many times to split each triangle.
    random_latent_size = latent_size #  MLP output layer size (embedding)
    random_gnn_msg_steps = msg_steps
    n_hidden_layers = 1
    lat_data = train_data.lat
    lon_data = train_data.lon
    divisions = 2
    # random_levels = 13
    resolution = train_data.lon.data.ptp() / train_data.lon.size
   
    if source == "Random":
        # Configurar parámetros para el modelo aleatorio.
        # if i == 0:
        params = None  # Se llenará a continuación
        state = {}
        model_config = graphcast_newvars.ModelConfig(
            resolution=resolution,
            mesh_size=random_mesh_size,
            latent_size=random_latent_size,
            gnn_msg_steps=random_gnn_msg_steps,
            hidden_layers=n_hidden_layers, #1,
            radius_query_fraction_edge_length=0.6,
            mesh2grid_edge_normalization_factor=None, #0.6180338738074472
            lat_data=lat_data,  
            lon_data=lon_data,
            divisions=divisions, 
            )
    # Devolver la configuración del modelo.
    
    return (model_config, params, state)

# Train step
def train_step(params: dict, grads: dict, learning_rate: float,
               ) -> tuple[dict, float]:

    if next(iter(next(iter(params.values())).values())).shape[0] == 8:
        flat_params, tree_params = jax.tree_util.tree_flatten(params)
        take_first_param = [param[0] for param in flat_params]
        params = jax.tree_util.tree_unflatten(tree_params, take_first_param)
    if next(iter(next(iter(grads.values())).values())).shape[0] == 8:
        flat_grads, tree_grads = jax.tree_util.tree_flatten(grads)
        sum_grads = [jnp.sum(g, axis=0) for g in flat_grads]
        grads = jax.tree_util.tree_unflatten(tree_grads, sum_grads)
    lr = learning_rate
    solver = optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-08, 
                         eps_root=0.0, mu_dtype=None,
                         weight_decay=0.1)
    optimizer = optax.chain(optax.clip_by_global_norm(32), solver)
    opt_state = optimizer.init(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
        
    return (params, lr)

train_step_jitted = jax.jit(train_step, device=jax.devices(backend='cpu')[0])

# Compilar la transformación run_forward con Haiku, JAX JIT y las configuraciones del modelo y la tarea.
load_data = data_load.Loader(n_samples=12, min_batch_size=8)
SST_SAT_with_nan = load_data.SST_SAT_with_nan
mean_by_level, stddev_by_level, diffs_stddev_by_level = load_data.load_norms()

def get_w_step(ds: xr.Dataset, var_name: str, v_min: float):
    sample_SST_SAT = ds#.isel(time=0)
    nan_mask = np.isnan(ds[var_name])#.isel(time=0))
    w_matrix_d = np.ceil(distance_transform_edt(~nan_mask, sampling=1))
    w_matrix_dinv = (w_matrix_d.max() - w_matrix_d) + 1
    w_step = np.where(w_matrix_dinv < (w_matrix_dinv.max() - (200 / 5.55 )) , v_min, 1)
    land_mask = np.where(np.isnan(sample_SST_SAT[var_name].data), 0, 1)
    m = w_step * land_mask

    return m / np.mean(m) # To normalize and cancel G

def get_w_lineal(ds: xr.Dataset, var_name: str,):
    (ds := ds.isel(time=0)) if 'time' in ds.dims else (ds := ds)
    sample_SST_SAT = ds
    nan_mask = np.isnan(ds[var_name])#.isel(time=0))
    w_matrix_d = np.ceil(distance_transform_edt(~nan_mask, sampling=1))
    w_matrix_dinv = (w_matrix_d.max() - w_matrix_d) + 1
    w_matrix_dinv_copy = w_matrix_dinv.copy()
    w_matrix_dinv_copy[w_matrix_dinv_copy < (w_matrix_dinv.max() - (200 / 5.55))] = w_matrix_dinv.max() - (200 / 5.55 )
    w_lineal = w_matrix_dinv_copy / np.mean(w_matrix_dinv_copy)
    land_mask = np.where(np.isnan(sample_SST_SAT[var_name].data), 0, 1)
    m = w_lineal * land_mask

    return m / np.mean(m) # To normalize and cancel G

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" 
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".10"

# %env XLA_PYTHON_CLIENT_PREALLOCATE=true
# %env XLA_PYTHON_CLIENT_ALLOCATOR=platform
# %env XLA_PYTHON_CLIENT_MEM_FRACTION=.50
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    )

# Phase 2: cos decay train 
user_linux, user_win = 'your_user', 'your_user'
# Logs file
path_logs = "path\\logs\\"
path_logs_linux = f'/home/{user_linux}/Doctorado/logs/'


def main(grid_hyperparam: dict, mask_filter: str, fast_test: bool, n_test: int):
    fast_test = fast_test 
    n_nb_test = n_test
    hyper_results = {}
    n_devices = len(jax.devices())
    if mask_filter == 'None': # None as str due to bash input
        weights = None #xr.concat([w_step] * 8, dim='device') # w_step
    elif mask_filter == 'lineal':
        weights = get_w_lineal(SST_SAT_with_nan, 'analysed_sst',)
        weights = xr.DataArray(weights, dims=['lat', 'lon'], 
                               coords={'lat': SST_SAT_with_nan['latitude'].data, 
                                       'lon': SST_SAT_with_nan['longitude'].data}
                              ).to_dataset(name='analysed_sst')
        weights = xr.concat([weights] * n_devices, dim='device')
    elif mask_filter == 'step':
        weights = get_w_step(SST_SAT_with_nan, 'analysed_sst', 0.1,)
        weights = xr.DataArray(weights, dims=['lat', 'lon'], 
                               coords={'lat': SST_SAT_with_nan['latitude'].data, 
                                       'lon': SST_SAT_with_nan['longitude'].data}
                              ).to_dataset(name='analysed_sst')
        weights = xr.concat([weights] * n_devices, dim='device')
    else:
        raise ValueError("Invalid mask_filter: must be 'None', 'lineal', or 'step'")
    global json_path 
    json_path = (path_logs_linux if sys.platform.startswith("linux") else path_logs) + f"results_{n_nb_test}.json"
    f = open((path_logs_linux if sys.platform.startswith("linux") else path_logs) 
             + f"logs_train_{n_nb_test}nb.txt", "w")
    # Pmap grads
    @functools.partial(xarray_jax_newvars.pmap, dim='device', in_axes=((0,) * 118) + ((0,) * 11) + (() if weights is None else (0,))) #---> if weights
    def pmap_grads(params, state, inputs, targets_template, forcings, weights):

        def with_params(fn):
            """Envuelve una función con los parámetros y el estado."""
            return functools.partial(fn, params=params, state=state)

        grads_fn_jitted = with_params(with_configs(grads_fn))

        loss_B, diagnostics, next_state, grads = grads_fn_jitted(
                                                    inputs=inputs,
                                                    targets=targets_template,
                                                    forcings=forcings,
                                                    weights=weights)
        return loss_B, diagnostics, next_state, grads
    
    # Pmap loss
    @functools.partial(xarray_jax_newvars.pmap, dim='device', in_axes=((0,) * 118) + (None,) +  ((0,) * 11) + (() if weights is None else (0,))) #---> if weights
    def pmap_loss(params, state, rng, inputs, targets_template, forcings, weights):

        def with_params(fn):
            """Envuelve una función con los parámetros y el estado."""
            return functools.partial(fn, params=params, state=state)


        def drop_state(fn):
            """Elimina el estado de salida de una función."""
            return lambda **kw: fn(**kw)[0]
        
        loss_fn_jitted = drop_state(with_params(with_configs(loss_fn.apply)))

        loss, diagnostics = loss_fn_jitted(
            rng=rng,
            inputs=inputs,
            targets=targets_template,
            forcings=forcings,
            weights=weights
            )
        return loss, diagnostics 

    # Pmap apply forward
    @functools.partial(xarray_jax_newvars.pmap, dim='device', in_axes=((0,) * 118) + (None,) +  ((0,) * 11))
    def pmap_forward(params, state, rng, inputs, targets_template, forcings):
        #args = tuple(kwargs.values())

        def with_params(fn):
            """Envuelve una función con los parámetros y el estado."""
            return functools.partial(fn, params=params, state=state)

        def drop_state(fn):
            """Elimina el estado de salida de una función."""
            return lambda **kw: fn(**kw)[0]
        run_forward_jitted = drop_state(with_params(with_configs(run_forward.apply)))

        predictions = rollout.chunked_prediction( # pmap_forward(# 
            run_forward_jitted,#(params, state), # run_forward_jitted_pmap,
            rng=rng, # jax.random.PRNGKey(0),
            inputs=inputs, # val_inputs,
            targets_template=targets_template, # val_targets * np.nan,
            forcings=forcings, # val_forcings
            )
        return predictions 

    print('FAST TEST') if fast_test else print('TRAINING')
    for cycle, M_i, msg_steps, latent_size, batch_size in itertools.product(*grid_hyperparam.values()):
        params_path = ((path_logs_linux if sys.platform.startswith("linux") else path_logs) 
                            + f"params_{n_nb_test}_cycle-{cycle}_M_i-{M_i}_msg_steps-{msg_steps}_latent_size-{latent_size}_batch_size-{batch_size}.npz") 
        try:
            # Write hyperparams
            print(f"Hyperparams: {cycle=}, {M_i=}, {msg_steps=}, {latent_size=}, {batch_size=}")
            f.write(f"Hyperparams: {cycle=}, {M_i=}, {msg_steps=}, {latent_size=}, {batch_size=}\n")
            # Empty results
            hist_loss_train = {'loss': [], 'hist': {}}
            hist_loss_val = {'loss': [], 'hist': {}}
            # Metrics
            hist_RMSE_val = {'wRMSE': {'mean': [], 'std': []}, 
                            'RMSE': {'mean': [], 'std': []}, 
                            'hist_wRMSE': {}, 
                            'hist_RMSE': {}}
            hist_ACC_val = []
            # Test results
            hist_loss_test = {'loss': [], 'hist': []}
            hist_RMSE_test = {'wRMSE': {'mean': [], 'std': []}, 
                            'RMSE': {'mean': [], 'std': []}, 
                            'hist_wRMSE': {'mean': [], 'std': []}, 
                            'hist_RMSE': {'mean': [], 'std': []}
                            }
            # If out-of-memory error
            out_of_memory = []
            # Model config due to the hyperparams here params out as None
            train_batch_gen, val_batch_gen, test_batch_gen = load_data()
            train_example_batch = load_data.train_example_batch
            model_config, params, state = data_model_config(cycle=cycle, M_i=M_i,
                                                msg_steps=msg_steps, 
                                                latent_size=latent_size, 
                                                train_data=train_example_batch)
            # Task config due to t-1, t
            n_days_input = 2
            time_unit = 'h'
            n_time = 24
            train_batch_size = len(list(train_batch_gen))
            val_batch_size = len(list(val_batch_gen))
            test_batch_size = len(list(test_batch_gen))

            task_config = graphcast_newvars.TaskConfig(
                        input_variables=graphcast_newvars.TASK_SST_NO_FORCING.input_variables,
                        target_variables=graphcast_newvars.TASK_SST_NO_FORCING.target_variables,
                        forcing_variables=graphcast_newvars.TASK_SST_NO_FORCING.forcing_variables,
                        pressure_levels=graphcast_newvars.PRESSURE_LEVELS_SST,
                        input_duration=graphcast_newvars.TASK_SST_NO_FORCING.input_duration
                        )
            def with_configs(fn):
                """Envuelve una función con las configuraciones del modelo y la tarea."""
                return functools.partial(
                    fn, model_config=model_config, task_config=task_config)
            init_jitted = jax.jit(with_configs(run_forward.init))
            # Pmap init forward
            @functools.partial(xarray_jax_newvars.pmap, dim='device', in_axes=(None,) + ((0,) * 11))
            def pmap_init(rng,
                        inputs,
                        targets_template,
                        forcings):

                params, state = init_jitted(
                    rng=rng,
                    inputs=inputs,
                    targets_template=targets_template,
                    forcings=forcings)
                    
                return params, state

            print(model_config)
            f.write(f"{model_config=}\n")
            # Train schedule
            n_devices = len(jax.devices())
            residual_iter = 1 if (train_batch_size % n_devices) != 0 else 0  
            iter_per_epoch = ((train_batch_size // n_devices) + residual_iter)
            transition_steps_cos = round(iter_per_epoch * cycle)
            cosine_schedule = optax.cosine_decay_schedule(1e-3, transition_steps_cos, alpha=0.0, exponent=1.0)
            # autoregressive steps schedule
            # n_aut_steps = train_example_batch.dims['time'] - n_days_input
            # autoregressive_step_schedule = gen_autoregressive_step_schedule(transition_steps_cos, n_aut_steps)
            global_it = 0
            # Fill 
            #for e in range(2): # Fast test
            for e in range(2) if fast_test else range(cycle): # range(2):
                try:
                    # Train:
                    # Loss train each epoch
                    hist_loss_train['hist'][f'{e}_epoch'] = []

                    # Validation:
                    # Loss for validation
                    hist_loss_val['hist'][f'{e}_epoch'] = []
                    # RMSE for validation
                    hist_RMSE_val['hist_wRMSE'][f'{e}_epoch'] = {'mean': [], 'std': []}
                    hist_RMSE_val['hist_RMSE'][f'{e}_epoch'] = {'mean': [], 'std': []}

                    #params, opt_state, loss, diagnostics, lr = train_step(e, params, train_example_batch, 
                    #                                                      batch_size, n_days_input, 
                    #                                                      task_config) 
                    # Train step
                    print('\nSTARTING TRAINING')
                    print(f"batch size: {train_batch_size}")
                    # len_batch = train_example_batch.dims['batch'] // batch_size
                    # Set batch gen
                    train_batch_gen, _, _ = load_data()
                    #train_batch_gen = batch_generator(train_example_batch, min_batch_size=batch_size, gentype='linear')
                    print(datetime.now().isoformat(sep=" "))
                    print(f"\nEpoch: {e}, -------------------------------\nmin-batch (size: {batch_size}), \nhist loss: {hist_loss_train['loss']}, \nOOM Errors: {out_of_memory}")
                    f.write(datetime.now().isoformat(sep=" "))
                    f.write(f"\nEpoch: {e}, -------------------------------\nmin-batch (size: {batch_size}), \nhist loss: {hist_loss_train['loss']},\nOOM Errors: {out_of_memory}\n")
                    # Fill the f'{e}_epoch': [loss_B1, loss_B2, ..., loss_Bn]
                    #for i_train_batch in itertools.islice(train_batch_gen, 0, 2): # Fast test
                    #for _ in range(2) if fast_test else range(iter_per_epoch):
                    for _ in range(2) if fast_test else range(iter_per_epoch):

                        # Trace number of iterations across epochs
                        #i += len_batch * e
                        # t+1 number of days ahead
                        train_steps = 1 #i_train_batch.sizes['time'] - n_days_input #12
                        # Handle the residual batch
                        try:
                            device_batch_data_train = {f'device_{i}': data_utils_newvars.extract_inputs_targets_forcings(
                                            next(train_batch_gen), 
                                            target_lead_times=slice(f"{n_time}{time_unit}", f"{train_steps * n_time}{time_unit}"),
                                            **dataclasses.asdict(task_config)) 
                                        for i in range(n_devices)}
                            train_inputs, train_targets, train_forcings = [xr.concat([device_batch_data_train[f'device_{i}'][j] 
                                                                    for i in range(n_devices)],
                                                                dim='device') 
                                                        for j in range(3)]
                        
                            # Compilar la inicialización de la transformación run_forward con Haiku y JAX JIT.
                            if params is None:
                                    params, state = pmap_init(
                                            jax.random.PRNGKey(0),# jax.random.split(jax.random.PRNGKey(0), 8), #
                                            train_inputs,
                                            train_targets,
                                            train_forcings)
                            assert len(next(iter(next(iter(params.values())).values())).sharding.device_set) == 8, 'Params are not in the 8 GPU'
                            # Calcular la pérdida utilizando la función de pérdida autoregresiva compilada
                            loss_B, diagnostics, next_state, grads = pmap_grads(
                                                                        params, state,
                                                                        train_inputs,
                                                                        train_targets,
                                                                        train_forcings,
                                                                        weights)
                            hist_loss_train['hist'][f'{e}_epoch'].append(loss_B.tolist())
                            # Fill loss per batch train: loss_e1 = [loss_B1, loss_B2, ...,  loss_Bn]
                            # Check if array is in GPU or CPU
                            # Any array mapped to GPU must be class jaxlib.xla_extension.ArrayImpl
                            # in the case of comes from cpu it should be numpy array
                            assert len(next(iter(next(iter(params.values())).values())).sharding.device_set) == 8, 'Params are not in GPU'
                            params = jax.device_get(params)
                            assert type(next(iter(next(iter(params.values())).values()))) == np.ndarray, 'Params are not in CPU'
                            grads = jax.device_get(grads)
                            lr = cosine_schedule(global_it)
                            params, lr = train_step_jitted(params, grads, lr)
                            del grads

                            # Checkpoint: save the params
                            ckpt = graphcast_newvars.CheckPoint(params=params,
                                    model_config=model_config,
                                    task_config=task_config,
                                    description=f'test_{n_nb_test}',
                                    license='license',
                                    )
                            with open(params_path, 'wb') as fp:
                                checkpoint.dump(fp, ckpt) 

                        except ValueError as exc:
                            if 'batch' in str(exc):
                                print('¡Residual train batch reached!')
                                lr = cosine_schedule(global_it)
                                # For residual case
                                assert len(next(iter(next(iter(params.values())).values())).sharding.device_set) == 8, 'Params are not in GPU'
                                params = jax.device_get(params)
                                assert type(next(iter(next(iter(params.values())).values()))) == np.ndarray, 'Params are not in CPU'
                                flat_params, tree_params = jax.tree_util.tree_flatten(params)
                                del params
                                take_first_param = [param[0] for param in flat_params]
                                params = jax.tree_util.tree_unflatten(tree_params, take_first_param)
                                del flat_params, take_first_param
                                def with_params(fn):
                                    """Envuelve una función con los parámetros y el estado."""
                                    return functools.partial(fn, params=params, state=state)
                                    
                                grads_fn_jitted = (with_params(with_configs(grads_fn)))
                                                                    #, 
                                                                    #device=jax.devices(backend='cpu')[0])))
                                residual_batch_loss = []
                                for i in range(n_devices):
                                    train_inputs, train_targets, train_forcings  = device_batch_data_train[f'device_{i}']
                                    loss_B, diagnostics, next_state, grads = grads_fn_jitted(inputs=jax.device_put(train_inputs, jax.devices(backend='cpu')[0]),
                                                                                targets=jax.device_put(train_targets, jax.devices(backend='cpu')[0]),
                                                                                forcings=jax.device_put(train_forcings, jax.devices(backend='cpu')[0]),
                                                                                weights=jax.device_put(weights.sel(device=0), jax.devices(backend='cpu')[0]) if (weights != None) else weights
                                                                                )
                                    residual_batch_loss.append(float(loss_B))
                                    params, lr = train_step_jitted(params, grads, lr)
                                    del grads
                                    ckpt = graphcast_newvars.CheckPoint(params=params,
                                                                model_config=model_config,
                                                                task_config=task_config,
                                                                description=f'test_{n_nb_test}_fine-tuning',
                                                                license='license',
                                                                )
                                    with open(params_path, 'wb') as fp:
                                        checkpoint.dump(fp, ckpt) 
                                hist_loss_train['hist'][f'{e}_epoch'].append(residual_batch_loss)
                                
                            else:
                                print(datetime.now().isoformat(sep=" "))
                                print(f' in epoch {e}, line: {sys.exc_info()[-1].tb_lineno} ' + str(exc))

                        print(f'\tcosine decay schedule: lr={lr}')
                        f.write(f'\tcosine decay schedule: lr={lr}\n')
                        # Imprimir la pérdida calculada
                        print(datetime.now().isoformat(sep=" "))
                        print(f"\titeration: {global_it}, \n\tmin-batch (size: {device_batch_data_train['device_0'][0].dims['batch']}), loss each device: {hist_loss_train['hist'][f'{e}_epoch']}")
                        f.write(datetime.now().isoformat(sep=" "))
                        f.write(f"\titeration: {global_it}, \n\tmin-batch (size: {device_batch_data_train['device_0'][0].dims['batch']}), loss each device: {hist_loss_train['hist'][f'{e}_epoch']}\n")
                        global_it += 1
                        params = jax.device_put_replicated(params, jax.local_devices())

                    # Fill loss per epoch loss_hyper = [e1_loss, e2_loss, ...,  en_loss]
                    hist_loss_train['loss'].append(np.mean(hist_loss_train['hist'][f'{e}_epoch']))
                                    
                    # climatology = # climatology by doy
                    # ACC_level = metrics.ACC(climatology).compute(forecast, truth)
                    #hist_ACC_train.append(ACC_level['analysed_sst'].data.mean())  
                    
                    # Validation step
                    print('\nSTARTING VALIDATION')
                    print(f"batch size: {val_batch_size}")
                    _, val_batch_gen, _ = load_data()
                    # val_batch_gen = batch_generator(val_example_batch, min_batch_size=batch_size, gentype='linear')
                    residual_val = val_batch_size % n_devices
                    val_iters_per_device = (val_batch_size // n_devices) + residual_val
                    #for i_val_batch in itertools.islice(val_batch_gen, 0, 2): # Fast test
                    for device_batch in itertools.islice(val_batch_gen, 0, 2) if fast_test else range(val_iters_per_device):#val_batch_gen:
                        if device_batch < (val_iters_per_device - 1):
                            val_steps = train_steps #i_val_batch.sizes['time'] - n_days_input 
                            device_batch_data_val = {f'device_{i}': data_utils_newvars.extract_inputs_targets_forcings(
                                            next(val_batch_gen), 
                                            target_lead_times=slice(f"{n_time}{time_unit}", f"{val_steps * n_time}{time_unit}"),
                                            **dataclasses.asdict(task_config)) 
                                        for i in range(n_devices)} 

                            val_inputs, val_targets, val_forcings = [xr.concat([device_batch_data_val[f'device_{i}'][j] 
                                                                for i in range(n_devices)],
                                                        dim='device') 
                                                    for j in range(3)]
                            
                            loss_val_B, diagnostics_val = pmap_loss(params, state, 
                                                                jax.random.PRNGKey(0),
                                                                val_inputs,
                                                                val_targets,
                                                                val_forcings,
                                                                weights)
                            hist_loss_val['hist'][f'{e}_epoch'].append(loss_val_B.tolist())
                            # Fill val loss per epoch loss_hyper = [e1_val_loss, e2_val_loss, ...,  en_val_loss]
                            predictions = pmap_forward(params, state, 
                                                    jax.random.PRNGKey(0),
                                                    val_inputs,
                                                    val_targets * np.nan,
                                                    val_forcings)
                        else:
                            print('¡Residual validation batch reached!')
                            residual_preds = []
                            residual_val_targets = []
                            for _ in range(residual_val):
                                val_inputs, val_targets, val_forcings = data_utils_newvars.extract_inputs_targets_forcings(
                                                                                next(val_batch_gen), 
                                                                                target_lead_times=slice(f"{n_time}{time_unit}", f"{val_steps * n_time}{time_unit}"),
                                                                                **dataclasses.asdict(task_config))
                                residual_val_targets.append(val_targets)
                                assert len(next(iter(next(iter(params.values())).values())).sharding.device_set) == 8, 'Params are not in GPU'
                                params = jax.device_get(params)
                                assert type(next(iter(next(iter(params.values())).values()))) == np.ndarray, 'Params are not in CPU'
                                flat_params, tree_params = jax.tree_util.tree_flatten(params)
                                del params
                                take_first_param = [param[0] for param in flat_params]
                                params = jax.tree_util.tree_unflatten(tree_params, take_first_param)
                                del flat_params, take_first_param

                                def with_params(fn):
                                    """Envuelve una función con los parámetros y el estado."""
                                    return functools.partial(fn, params=params, state=state)
                                def drop_state(fn):
                                    """Elimina el estado de salida de una función."""
                                    return lambda **kw: fn(**kw)[0]
                                
                                loss_fn_jitted = drop_state(with_params(with_configs(loss_fn.apply)))

                                loss_val_B, diagnostics_val = loss_fn_jitted(
                                    rng=jax.random.PRNGKey(0),
                                    inputs=jax.device_put(val_inputs, jax.devices(backend='cpu')[0]),
                                    targets=jax.device_put(val_targets, jax.devices(backend='cpu')[0]),
                                    forcings=jax.device_put(val_forcings, jax.devices(backend='cpu')[0]),
                                    weights=jax.device_put(weights.sel(device=0), jax.devices(backend='cpu')[0]) if (weights != None) else weights
                                    )
                                hist_loss_val['hist'][f'{e}_epoch'].append(float(loss_val_B))
                                # Fill val loss per epoch loss_hyper = [e1_val_loss, e2_val_loss, ...,  en_val_loss]
                                def with_params(fn):
                                    """Envuelve una función con los parámetros y el estado."""
                                    return functools.partial(fn, params=params, state=state)
                                def drop_state(fn):
                                    """Elimina el estado de salida de una función."""
                                    return lambda **kw: fn(**kw)[0]
                                run_forward_jitted = drop_state(with_params(with_configs(run_forward.apply)))

                                #run_forward_jitted_pmap = xarray_jax_newvars.pmap(run_forward_jitted, 
                                #                                                dim='device', 
                                #                                                in_axes=(None,) + ((0,) * 11))
                                predictions = rollout.chunked_prediction( # pmap_forward(# 
                                    run_forward_jitted,#(params, state), # run_forward_jitted_pmap,
                                    rng=jax.random.PRNGKey(0), # jax.random.PRNGKey(0),
                                    inputs=jax.device_put(val_inputs, jax.devices(backend='cpu')[0]), # val_inputs,
                                    targets_template=jax.device_put(val_targets * np.nan, jax.devices(backend='cpu')[0]), # val_targets * np.nan,
                                    forcings=jax.device_put(val_forcings, jax.devices(backend='cpu')[0]), # val_forcings
                                    )
                                residual_preds.append(predictions)
                                
                            predictions = xr.concat(residual_preds, dim='device')
                            val_targets = xr.concat(residual_val_targets, dim='device')
                            params = jax.device_put_replicated(params, jax.local_devices()) 
                        # Metrics for validation
                        # wRMSE_level_val = metrics.RMSE(predictions, val_targets, weights_type='cos')['analysed_sst'].data
                        land_mask = metrics.get_land_mask_batched(SST_SAT_with_nan['analysed_sst'], 
                                                                predictions['analysed_sst'].isel(time=0, 
                                                                                                device=0))
                        for n in range(residual_val) if (device_batch >= (val_iters_per_device - 1)) else range(n_devices):
                            wRMSE_mean_val, wRMSE_std_val = metrics.RMSE_mean_std(predictions['analysed_sst'].isel(device=n), 
                                                                                val_targets['analysed_sst'].isel(device=n),
                                                                                land_mask=land_mask,
                                                                                weights_type='cos')
                            hist_RMSE_val['hist_wRMSE'][f'{e}_epoch']['mean'].append(wRMSE_mean_val.to_numpy().tolist())
                            hist_RMSE_val['hist_wRMSE'][f'{e}_epoch']['std'].append(wRMSE_std_val.to_numpy().tolist())
                            
                            RMSE_mean_val, RMSE_std_val = metrics.RMSE_mean_std(predictions['analysed_sst'].isel(device=n), 
                                                                                val_targets['analysed_sst'].isel(device=n),
                                                                                land_mask=land_mask, 
                                                                                weights_type=None)
                            hist_RMSE_val['hist_RMSE'][f'{e}_epoch']['mean'].append(RMSE_mean_val.to_numpy().tolist())
                            hist_RMSE_val['hist_RMSE'][f'{e}_epoch']['std'].append(RMSE_std_val.to_numpy().tolist())

                    # Fill RMSE and wRMSE per epoch RMSE_hyper = [e1_RMSE, e2_RMSE, ...,  en_RMSE]
                    hist_loss_val['loss'].append(np.mean(list(itertools.chain.from_iterable(item if isinstance(item, list) else [item] 
                                                            for item in hist_loss_val['hist'][f'{e}_epoch']))))
                    
                    hist_RMSE_val['wRMSE']['mean'].extend(np.mean(hist_RMSE_val['hist_wRMSE'][f'{e}_epoch']['mean'], axis=0).tolist())
                    hist_RMSE_val['wRMSE']['std'].extend(np.mean(hist_RMSE_val['hist_wRMSE'][f'{e}_epoch']['std'], axis=0).tolist())
                    hist_RMSE_val['RMSE']['mean'].extend(np.mean(hist_RMSE_val['hist_RMSE'][f'{e}_epoch']['mean'], axis=0).tolist())
                    hist_RMSE_val['RMSE']['std'].extend(np.mean(hist_RMSE_val['hist_RMSE'][f'{e}_epoch']['std'], axis=0).tolist())

                    print(datetime.now().isoformat(sep=" "))
                    print(f"\nmin-batch (size: {device_batch_data_val['device_0'][0].dims['batch']}), \nhist val_loss each epoch: {hist_loss_val['loss']} \nwlatRMSE val: {np.nanmean(np.array(hist_RMSE_val['wRMSE']['mean']))}, RMSE val: {np.nanmean(np.array(hist_RMSE_val['RMSE']['mean']))} \nACC val: {hist_ACC_val}")
                    f.write(datetime.now().isoformat(sep=" "))
                    f.write(f"\nmin-batch (size: {device_batch_data_val['device_0'][0].dims['batch']}), \nhist val_loss each epoch: {hist_loss_val['loss']} \nwlatRMSE val: {np.nanmean(np.array(hist_RMSE_val['wRMSE']['mean']))}, RMSE val: {np.nanmean(np.array(hist_RMSE_val['RMSE']['mean']))} \nACC val: {hist_ACC_val}\n")
                except XlaRuntimeError as exc:
                    print(datetime.now().isoformat(sep=" "))
                    print(f' in epoch {e}, line: {sys.exc_info()[-1].tb_lineno} ' + str(exc))
                    if f"RESOURCE_EXHAUSTED" in str(exc):
                        print(datetime.now().isoformat(sep=" "))
                        print(f"Caught out-of-memory error! in epoch {e}: {cycle=}, {M_i=}, {msg_steps=}, {latent_size=}, {batch_size=}")
                        f.write(datetime.now().isoformat(sep=" "))
                        f.write(f"Caught out-of-memory error! in epoch {e}: {cycle=}, {M_i=}, {msg_steps=}, {latent_size=}, {batch_size=}\n")
                        #hyper_results[f"{cycle=}, {M_i=}, {msg_steps=}, {latent_size=}, {batch_size=}"]["ERROR"] = str(exc)
                        out_of_memory.append(str(exc) + f' in epoch {e}, line: {sys.exc_info()[-1].tb_lineno}')
            # Test metrics
            print('\nSTARTING TEST')
            print(f"batch size: {test_batch_size}")
            _, _, test_batch_gen = load_data()
            # test_batch_gen = batch_generator(test_example_batch, min_batch_size=batch_size, gentype='linear')
            residual_test = test_batch_size % n_devices
            test_iters_per_device = (test_batch_size // n_devices) + residual_test
            #for i_test_batch in itertools.islice(test_batch_gen, 0, 2): # Fast test
            #for i_test_batch in itertools.islice(test_batch_gen, 0, 2) if fast_test else test_batch_gen:
            for device_batch in itertools.islice(test_batch_gen, 0, 2) if fast_test else range(test_iters_per_device):
                if device_batch < (test_iters_per_device - 1):
                    test_steps = train_steps
                    
                    device_batch_data_test = {f'device_{i}': data_utils_newvars.extract_inputs_targets_forcings(
                                            next(test_batch_gen), 
                                            target_lead_times=slice(f"{n_time}{time_unit}", f"{test_steps * n_time}{time_unit}"),
                                            **dataclasses.asdict(task_config)) 
                                        for i in range(n_devices)}
                    test_inputs, test_targets, test_forcings = [xr.concat([device_batch_data_test[f'device_{i}'][j] 
                                                                for i in range(n_devices)],
                                                        dim='device') 
                                                    for j in range(3)]
                    loss_test_B, diagnostics_test = pmap_loss(params, state, 
                                                            jax.random.PRNGKey(0),
                                                            test_inputs,
                                                            test_targets,
                                                            test_forcings,
                                                            weights)
                    hist_loss_test['hist'].append(loss_test_B.tolist())
                    # loss_hyper = [e1_test_loss, e2_test_loss, ...,  en_test_loss]
                    predictions = pmap_forward(params, state, 
                                            jax.random.PRNGKey(0),
                                            test_inputs,
                                            test_targets * np.nan,
                                            test_forcings)
                else:
                    print('¡Residual test batch reached!')
                    residual_preds = []
                    residual_test_targets = []
                    for _ in range(residual_test):
                        test_inputs, test_targets, test_forcings = data_utils_newvars.extract_inputs_targets_forcings(
                                                                        next(test_batch_gen), 
                                                                        target_lead_times=slice(f"{n_time}{time_unit}", f"{test_steps * n_time}{time_unit}"),
                                                                        **dataclasses.asdict(task_config))
                        residual_test_targets.append(test_targets)
                        assert len(next(iter(next(iter(params.values())).values())).sharding.device_set) == 8, 'Params are not in GPU'
                        params = jax.device_get(params)
                        assert type(next(iter(next(iter(params.values())).values()))) == np.ndarray, 'Params are not in CPU'
                        flat_params, tree_params = jax.tree_util.tree_flatten(params)
                        del params
                        take_first_param = [param[0] for param in flat_params]
                        params = jax.tree_util.tree_unflatten(tree_params, take_first_param)
                        del flat_params, take_first_param

                        def with_params(fn):
                            """Envuelve una función con los parámetros y el estado."""
                            return functools.partial(fn, params=params, state=state)
                        def drop_state(fn):
                            """Elimina el estado de salida de una función."""
                            return lambda **kw: fn(**kw)[0]
                        
                        loss_fn_jitted = drop_state(with_params(with_configs(loss_fn.apply)))

                        loss_test_B, diagnostics_test = loss_fn_jitted(
                            rng=jax.random.PRNGKey(0),
                            inputs=jax.device_put(test_inputs, jax.devices(backend='cpu')[0]),
                            targets=jax.device_put(test_targets, jax.devices(backend='cpu')[0]),
                            forcings=jax.device_put(test_forcings, jax.devices(backend='cpu')[0]),
                            weights=jax.device_put(weights.sel(device=0), jax.devices(backend='cpu')[0]) if (weights != None) else weights
                            )
                        hist_loss_test['hist'].append(float(loss_test_B))
                        # Fill val loss per epoch loss_hyper = [e1_val_loss, e2_val_loss, ...,  en_val_loss]
                        def with_params(fn):
                            """Envuelve una función con los parámetros y el estado."""
                            return functools.partial(fn, params=params, state=state)
                        def drop_state(fn):
                            """Elimina el estado de salida de una función."""
                            return lambda **kw: fn(**kw)[0]
                        run_forward_jitted = drop_state(with_params(with_configs(run_forward.apply)))

                        #run_forward_jitted_pmap = xarray_jax_newvars.pmap(run_forward_jitted, 
                        #                                                dim='device', 
                        #                                                in_axes=(None,) + ((0,) * 11))
                        predictions = rollout.chunked_prediction( # pmap_forward(# 
                            run_forward_jitted,#(params, state), # run_forward_jitted_pmap,
                            rng=jax.random.PRNGKey(0), # jax.random.PRNGKey(0),
                            inputs=jax.device_put(test_inputs, jax.devices(backend='cpu')[0]), # val_inputs,
                            targets_template=jax.device_put(test_targets * np.nan, jax.devices(backend='cpu')[0]), # val_targets * np.nan,
                            forcings=jax.device_put(test_forcings, jax.devices(backend='cpu')[0]), # val_forcings
                            )
                        residual_preds.append(predictions)

                    predictions = xr.concat(residual_preds, dim='device')
                    test_targets = xr.concat(residual_test_targets, dim='device')
                # Metrics for test
                land_mask = metrics.get_land_mask_batched(SST_SAT_with_nan['analysed_sst'], 
                                                                predictions['analysed_sst'].isel(time=0, 
                                                                                                device=0))
                for n in range(residual_test) if (device_batch >= (test_iters_per_device - 1)) else range(n_devices):           
                    wRMSE_mean_test, wRMSE_std_test = metrics.RMSE_mean_std(predictions['analysed_sst'].isel(device=n), 
                                                                            test_targets['analysed_sst'].isel(device=n),
                                                                            land_mask=land_mask,
                                                                            weights_type='cos')
                    
                    hist_RMSE_test['hist_wRMSE']['mean'].append(wRMSE_mean_test.to_numpy().tolist())
                    hist_RMSE_test['hist_wRMSE']['std'].append(wRMSE_std_test.to_numpy().tolist())

                    RMSE_mean_test, RMSE_std_test = metrics.RMSE_mean_std(predictions['analysed_sst'].isel(device=n), 
                                                                        test_targets['analysed_sst'].isel(device=n),
                                                                        land_mask=land_mask, 
                                                                        weights_type=None)
                    
                    hist_RMSE_test['hist_RMSE']['mean'].append(RMSE_mean_test.to_numpy().tolist())
                    hist_RMSE_test['hist_RMSE']['std'].append(RMSE_std_test.to_numpy().tolist())
    
            # Fill RMSE and wRMSE per epoch RMSE_hyper = [e1_RMSE, e2_RMSE, ...,  en_RMSE]
            hist_loss_test['loss'].append(np.mean(list(itertools.chain.from_iterable(item if isinstance(item, list) else [item] 
                                                            for item in hist_loss_test['hist']))))
                                                            
            hist_RMSE_test['wRMSE']['mean'].append(np.mean(hist_RMSE_test['hist_wRMSE']['mean'], axis=0).tolist())
            hist_RMSE_test['wRMSE']['std'].append(np.mean(hist_RMSE_test['hist_wRMSE']['std'], axis=0).tolist())
            hist_RMSE_test['RMSE']['mean'].append(np.mean(hist_RMSE_test['hist_RMSE']['mean'], axis=0).tolist())
            hist_RMSE_test['RMSE']['std'].append(np.mean(hist_RMSE_test['hist_RMSE']['std'], axis=0).tolist())

            print(datetime.now().isoformat(sep=" "))
            print(f"\nmin-batch (size: {device_batch_data_test['device_0'][0].dims['batch']}), \ntest_loss: {hist_loss_test['loss'][0]} \nwlatRMSE test: {np.nanmean(np.array(hist_RMSE_test['wRMSE']['mean']))}, RMSE test: {np.nanmean(np.array(hist_RMSE_test['RMSE']['mean']))} \nACC test: ")
            f.write(datetime.now().isoformat(sep=" "))
            f.write(f"\nmin-batch (size: {device_batch_data_test['device_0'][0].dims['batch']}), \ntest_loss: {hist_loss_test['loss'][0]} \nwlatRMSE test: {np.nanmean(np.array(hist_RMSE_test['wRMSE']['mean']))}, RMSE test: {np.nanmean(np.array(hist_RMSE_test['RMSE']['mean']))} \nACC test: \n")
        
        except XlaRuntimeError as exc:
            print(datetime.now().isoformat(sep=" "))
            print(f' in Test, line: {sys.exc_info()[-1].tb_lineno} ' + str(exc))
            if "RESOURCE_EXHAUSTED" in str(exc):
                print(datetime.now().isoformat(sep=" "))
                print(f"Caught out-of-memory error!: {cycle=}, {M_i=}, {msg_steps=}, {latent_size=}, {batch_size=}")
                f.write(datetime.now().isoformat(sep=" "))
                f.write(f"Caught out-of-memory error!: {cycle=}, {M_i=}, {msg_steps=}, {latent_size=}, {batch_size=}\n")
                out_of_memory.append(str(exc)+ f' in epoch test, line: {sys.exc_info()[-1].tb_lineno}')
        
        hyper_results[f"{cycle=}, {M_i=}, {msg_steps=}, {latent_size=}, {batch_size=}"] = {'hist_loss_train': hist_loss_train,
                                                                                        'hist_loss_val': hist_loss_val,
                                                                                        'hist_RMSE_val': hist_RMSE_val,
                                                                                        'loss_test': hist_loss_test,
                                                                                        'RMSE_test': hist_RMSE_test,
                                                                                        'OOM_ERROR': out_of_memory}
        # Save results
        # with open(json_path, "w") as js:
        #    # Write the dictionary to the file
        #    json.dump(hyper_results, js, indent=4)
    f.close()

    destination = f'/home/{user_linux}/Doctorado/params/GraphCast'
    shutil.copy(params_path, destination)

    return hyper_results

if __name__ == "__main__":
    # Hyper param Grid serach
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("n_epochs", 
                        help="number of epochs e.g. 10",
                        type=int)
    parser.add_argument("M_i", 
                        help="number of mesh refinaments e.g. 3",
                        type=int)
    parser.add_argument("msg_steps", 
                        help="number of message passing steps e.g. 4",
                        type=int)
    parser.add_argument("latent_size", 
                        help="size of the MLP e.g. 32",
                        type=int)
    parser.add_argument("batch_size", 
                        help="batch size e.g. 8",
                        type=int)
    parser.add_argument("mask_filter", 
                        help="define mask_filter for the loss functions e.g. 'None', 'lineal', or 'step'",
                        type=str)
    parser.add_argument("-ft", "--fast_test", action='store_true',
                    help="if True it performs a short demo of the training")
    parser.add_argument("n_test", 
                        help="save the outputs with this number e.g. 41",
                        type=int)
    
    args = parser.parse_args()
    grid_hyperparam = {"n_cycles": [args.n_epochs],#[10],#[50],
                       "M_i": [args.M_i],#[3],
                       "msg_steps": [args.msg_steps],#[4],
                       "latent_size": [args.latent_size],#[32],
                       "batch_size": [args.batch_size],#[8], #[16],
                      }                    
    hyper_results = main(grid_hyperparam, args.mask_filter, args.fast_test, args.n_test)
    # Save results
    with open(json_path, "w") as js:
        # Write the dictionary to the file
        json.dump(hyper_results, js, indent=4)