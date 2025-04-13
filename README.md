# Adan

**Overview**:

The **Adan (Adaptive Nesterov Momentum)** optimizer is a next-generation optimization algorithm designed to accelerate training and improve convergence in deep learning models. It combines **adaptive gradient estimation** and **multi-step momentum** for enhanced performance.

This algorithm is introduced in the paper:
- **"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"** ([arXiv link](https://arxiv.org/abs/2208.06677)).

The implementation is inspired by the official repository:
- [Adan GitHub Repository](https://github.com/sail-sg/Adan)

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: Learning rate for the optimizer.
- **`beta1`** *(float, default=0.98)*: Exponential decay rate for the first moment estimates.
- **`beta2`** *(float, default=0.92)*: Exponential decay rate for gradient difference momentum.
- **`beta3`** *(float, default=0.99)*: Exponential decay rate for the second moment estimates.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability.
- **`weight_decay`** *(float, default=0.0)*: Strength of weight decay regularization.
- **`no_prox`** *(bool, default=False)*: If `True`, disables proximal updates during weight decay.
- **`foreach`** *(bool, default=True)*: Enables multi-tensor operations for optimization.
- **`clipnorm`** *(float, optional)*: Clips gradients by their norm.
- **`clipvalue`** *(float, optional)*: Clips gradients by their value.
- **`global_clipnorm`** *(float, optional)*: Clips gradients by their global norm.
- **`use_ema`** *(bool, default=False)*: Enables Exponential Moving Average (EMA) for model parameters.
- **`ema_momentum`** *(float, default=0.99)*: EMA momentum for parameter averaging.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting model parameters with EMA values.
- **`loss_scale_factor`** *(float, optional)*: Scaling factor for loss values in mixed-precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: Number of steps for gradient accumulation.
- **`name`** *(str, default="adan")*: Name of the optimizer.

---

**Example Usage**:

```python
import tensorflow as tf
from adan import Adan

# Initialize the Adan optimizer
optimizer = Adan(
    learning_rate=1e-3,
    beta1=0.98,
    beta2=0.92,
    beta3=0.99,
    weight_decay=0.01,
    use_ema=True,
    ema_momentum=0.999
)

# Compile a model
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```
