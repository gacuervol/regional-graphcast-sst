import jax
import numpy as np
import xarray as xr
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from typing import Tuple
## Progress bar
from tqdm.auto import tqdm

class ConvLSTMBlock(nn.Module):
    """Convolutional LSTM block."""
    carry_rng: jax.random.PRNGKey
    input_shape: tuple[int, int, int, int, int]
    
    def setup(self):
        
        self.convlstm1 = nn.ConvLSTMCell(features=8, kernel_size=(3,3), strides=1, padding=1)
        self.convlstm2 = nn.ConvLSTMCell(features=1, kernel_size=(3,3), strides=1, padding=1)
        self.carry = self.initialize_carry(self.carry_rng, input_shape=self.input_shape)
        
    def __call__(self, x: .Array):

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

def xarr_zeros(shape: tuple) -> xr.DataArray:
    """Create a xarray DataArray filled with zeros"""

    b, t, h, w = shape
    da = xr.DataArray(
        # data=np.zeros([8, 2, 301, 329]),
        data=np.zeros(shape),
        dims=['batch', 'time', 'lat', 'lon'],
        coords=dict(
            lat=(["lat"], np.arange(19.57, 34.58, 0.05)),
            lon=(["lon"], np.arange(-20.93, -4.525, 0.05)),
            time=(
                ["time"], 
                np.arange(np.timedelta64(0, 'D'), np.timedelta64(t, 'D'))
                ),
            #batch=(["batch"], np.arange(8)),
            datetime=(
                ["batch", "time"], 
                np.arange(np.datetime64('2021-01-01'), np.datetime64(f'2021-01-{(b*t)+1}')).reshape(b, t)
                ),
            ),
        )

    return da

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

@jax.jit  # Jit the function for efficiency
def train_step(
    state: train_state.TrainState, 
    inputs_norm: jax.Array, 
    dx_target_norm: jax.Array
    )-> tuple[train_state.TrainState, jax.Array, jax.Array]:

    # Define the loss function
    def loss_fn(params):
        
        dx_predictions = state.apply_fn(params, inputs_norm)
        MSE = (dx_predictions - dx_target_norm) ** 2
        return jnp.mean(MSE)  # MSE loss

    # Gradient function
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # Update function
    state = state.apply_gradients(grads=grads)

    return state, loss, grads   

@jax.jit
def eval_step(
    state: train_state.TrainState, 
    inputs_norm: jax.Array, 
    dx_target_norm: jax.Array,
    )-> tuple[train_state.TrainState, jax.Array, jax.Array]:
    """Train for a single step."""
    
    dx_predictions_norm = state.apply_fn(state.params, inputs_norm)
    MSE = (dx_predictions_norm - dx_target_norm) ** 2
    loss = jnp.mean(MSE)  # MSE loss

    return loss, dx_predictions_norm

def eval_metrics(
    inputs_norm: jax.Array,
    targets_norm: jax.Array, 
    dx_predictions_norm: jax.Array, 
    normalizer: Normalizer,
    )-> dict:
    
    inputs = normalizer.unnormalize(inputs_norm)
    last_input = inputs.at[:, -1, :, :, :].get().squeeze()
    dx_predictions = normalizer.unnormalize_residual(dx_predictions_norm).squeeze()
    predictions = last_input + dx_predictions
    targets = normalizer.unnormalize(targets_norm).squeeze()
    RMSE = lambda f, o: jnp.sqrt(jnp.mean((f - o) ** 2))
    metrics = {
        'RMSE': RMSE(predictions, targets),
        }

    return metrics, predictions

def train_model(
    state, 
    data_path, 
    reshape_input, 
    norms_factors, 
    num_epochs, 
    t_steps,
    )-> tuple[train_state.TrainState, dict]:

    # Training loop
    metrics_history = {
        'train_loss': [],
        'validation_loss': [],
        'validation_RMSE': [],
        }

    for epoch in tqdm(range(num_epochs)):
        # Load train, val and test generators
        data_loader = data_load.Loader(
            n_samples=12, 
            min_batch_size=8, 
            data_path=data_path
            )
        train_gen, validation_gen, _ = data_loader()
        loss_sum = jnp.array([0.])
        steps = jnp.array([0.])
        print(f"Time range train: {data_loader.train_range}")
        
        for batch in train_gen:
            # Instantiate the normalizer
            normalizer = Normalizer(**norms_factors)
            # Normalize and reshape the batch
            inputs_norm, targets_norm, dx_target_norm = normalize_and_reshape_batch(
                batch, 
                normalizer, 
                t_steps, 
                reshape_input,
                )
            # Train the model
            state, loss, grads = train_step(
                state, 
                inputs_norm, 
                dx_target_norm,
                )
            loss_sum += loss
            steps += 1.0

        # print(grads)
        metrics_history['train_loss'].append(np.array(loss_sum / steps)[0])
        print(f"Epoch: {epoch}, train_loss: {metrics_history['train_loss'][-1]}")
        
        loss_sum = jnp.array([0.])
        RMSE_sum = jnp.array([0.])
        steps = jnp.array([0.])
        print(f"Time range validation: {data_loader.val_range}")

        for batch in validation_gen:
            # Normalize and reshape the batch
            inputs_norm, targets_norm, dx_target_norm = normalize_and_reshape_batch(
                batch, 
                normalizer, 
                t_steps, 
                reshape_input,
                )
            # Evaluate the model
            loss, dx_predictions_norm = eval_step(
                state, 
                inputs_norm, 
                dx_target_norm,
                )
            # Calculate the metrics
            metrics, _ = eval_metrics(
                inputs_norm,
                targets_norm, 
                dx_predictions_norm, 
                normalizer,
                )
            # cumulate the metrics
            loss_sum += loss
            RMSE_sum += metrics['RMSE']
            steps += 1.0
            
        metrics_history['validation_loss'].append(np.array(loss_sum / steps)[0])
        metrics_history['validation_RMSE'].append(np.array(RMSE_sum / steps)[0])
        print(f"Epoch: {epoch}, validation_loss: {metrics_history['validation_loss'][-1]}")
        print(f"Epoch: {epoch}, validation_RMSE: {jnp.mean(metrics_history['validation_RMSE'][-1])}")

    return state, metrics_history
