import jax
import numpy as np
import jax.numpy as jnp
# from flax.training.train_state import TrainState
# from flax.training import checkpoints
from flax.training import train_state
import optax
from .convlstm import ConvLSTMBlock, train_model

# Import local modules
import os, sys
folders_lis = os.getcwd().split(os.sep)
repo_path = os.sep.join(folders_lis[:folders_lis.index('PhD_repo') + 1])
if sys.platform.startswith("win"):
    sys.path.append(repo_path + '\\src')
else:
    sys.path.append(repo_path + '/src')

from art1_tools import data_load

# Convert inputs to JAX arrays
data_path = "~/repo/data/IBI_SST_L4_FULL.nc"
data_loader = data_load.Loader(
    n_samples=12, 
    min_batch_size=8, 
    data_path=data_path
    )
mean_by_level, stddev_by_level, diffs_stddev_by_level = data_loader.load_norms()
norms_factors = {
    'scales': jnp.array(stddev_by_level['analysed_sst'].data),
    'locations': jnp.array(mean_by_level['analysed_sst'].data),
    'residual_scales': jnp.array(diffs_stddev_by_level['analysed_sst'].data),
    }

# Instantiate the model
carry_rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 2)

input_shape = (8, 2, 256, 256, 1)
model = ConvLSTMBlock(carry_rng, input_shape)
# carry = model.initialize_carry(carry_rng, (8, 1, 500, 600, 1))

# Initialize the model to get the parameters
# init_rng = jax.random.key(1)
params = model.init(
    init_rng, 
    jnp.ones(input_shape),
    )
# Define the optimizer    
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
# Define the state: 
# to packs the parameters, the optimizer, and the forward step of the model 
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer,
    )
# Train the model
state_trained, hist = train_model(
    state, 
    data_path, 
    input_shape,
    norms_factors,
    num_epochs=150,
    t_steps=1,
    )
