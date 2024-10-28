import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad, jit
import optax
from typing import NamedTuple, Any


def loss_fn(params, model, batch):
    inputs, targets = batch
    predictions = model.apply(params, inputs)
    ## Should replace with sigmoid loss for image pairs later
    loss = jnp.mean((predictions - targets) ** 2)  
    return loss

@jit
def train_step(state: TrainingState, batch, model, opt_update):
    """Perform a single training step with parameter freezing for base weights."""
    
    loss, grads = value_and_grad(loss_fn)(state.params, model, batch)
    frozen_grads = jax.tree_map(lambda g, p: g if 'lora_A' in p or 'lora_B' in p else jnp.zeros_like(g), grads, state.params)
    updates, new_opt_state = opt_update(frozen_grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    return TrainingState(params=new_params, opt_state=new_opt_state, step=state.step + 1), loss

def create_optimizer(learning_rate=1e-4):
    optimizer = optax.adam(learning_rate)
    return optimizer

def train_model(model, train_data, num_steps=1000):
    """Main training loop for the mock model."""
    # Initialize parameters and optimizer
    key = random.PRNGKey(42)
    params = model.init(key, jnp.ones([32, 10]))  # Mock input size [batch_size, feature_dim]
    optimizer = create_optimizer()
    opt_state = optimizer.init(params)
    state = TrainingState(params=params, opt_state=opt_state, step=0)

    for step in range(num_steps):
        batch = next(train_data)
        state, loss = train_step(state, batch, model, optimizer.update)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss}")

    return state

def mock_data_generator(batch_size=32, input_dim=10, output_dim=128):
    """Yield mock data for training."""
    while True:
        inputs = jax.random.normal(random.PRNGKey(0), (batch_size, input_dim))
        targets = jax.random.normal(random.PRNGKey(1), (batch_size, output_dim))  
        yield inputs, targets

mock_model = MockModel()
mock_data = mock_data_generator()

trained_state = train_model(mock_model, mock_data)