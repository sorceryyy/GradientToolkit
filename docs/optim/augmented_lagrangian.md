### Method Introduction
The **Augmented Lagrangian Method (ALM)** is an advanced optimization technique designed to solve constrained nonlinear programming problems. By combining Lagrangian duality with quadratic penalty terms, it enhances numerical stability and convergence properties compared to classical approaches.

#### Key Components
1. **Augmented Lagrangian Function**:
   ```math
   \mathcal{L}_A(\mathbf{x}, \boldsymbol{\lambda}, \rho) = f(\mathbf{x}) + \boldsymbol{\lambda}^\top \mathbf{h}(\mathbf{x}) + \frac{\rho}{2}\|\mathbf{h}(\mathbf{x})\|^2
   ```
   where:
   - $f(\mathbf{x})$ is the original objective function
   - $\boldsymbol{\lambda}^\top \mathbf{h}(\mathbf{x})$ is the Lagrangian term for constraints
   - $\frac{\rho}{2}\|\mathbf{h}(\mathbf{x})\|^2$ is the quadratic penalty for constraint violations

2. **Dual Update Mechanism**:
   - Progressive multiplier update:
     ```math
     \boldsymbol{\lambda}^{k+1} = \boldsymbol{\lambda}^k + \rho \mathbf{h}(\mathbf{x}^k)
     ```
   - Adaptive penalty parameter $\rho$ controls constraint enforcement

#### Algorithmic Features
- Handles equality constraints directly and inequality constraints through slack variables
- Enables inexact minimization while maintaining convergence guarantees
- Reduces ill-conditioning issues common in pure penalty methods
- Suitable for both convex and non-convex problems
- Effective in distributed optimization settings

#### Advantages
- Provides a balanced approach between primal feasibility and dual optimality
- Achieves faster convergence compared to pure penalty methods
- Shows robustness to poor initial guesses through adaptive parameter tuning
- Serves as the foundation for modern variants like ADMM (Alternating Direction Method of Multipliers)

## Usage with pytorch lightning
you can train your model with following code:
```python
from gradient_utils.optim.schedulers import AugLagLRConfig, AugLagLR, AugLagLossCalculator
def test_solve_auglag():
    """
    Test that `AugLagLR` can solve the constrained optimization problem.

        min x² s.t. x > 3

    The solution being that x is approximately 3
    """
    x = Parameter(torch.zeros((), requires_grad=True))
    group_lr = {"x": 0.1}
    parameter_list = [{"params": x, "name": "x", "lr": group_lr["x"]}]

    optimizer = Adam(parameter_list)
    scheduler = AugLagLR(
        config=AugLagLRConfig(
            lr_update_lag=1,
            lr_update_lag_best=100,
            lr_init_dict=group_lr,
            aggregation_period=1,
            lr_factor=0.5,
            penalty_progress_rate=1.1,
            safety_rho=100,
            safety_alpha=100,
            max_lr_down=10,
            inner_early_stopping_patience=10,
            max_outer_steps=100000,
            patience_penalty_reached=10,
            patience_max_rho=10,
            penalty_tolerance=1e-5,
            max_inner_steps=10,
        )
    )

    auglag_loss = AugLagLossCalculator(init_alpha=0.0, init_rho=1.0)
    step_counter = 0
    max_iter = 10000
    constraint = torch.inf
    for _ in range(max_iter):
        optimizer.zero_grad()
        loss = x**2
        constraint = torch.max(3 - x, torch.zeros(()))
        auglag_loss_tensor = auglag_loss(loss, constraint)
        auglag_loss_tensor.backward()
        optimizer.step()
        converged = scheduler.step(
            optimizer=optimizer, loss=auglag_loss, loss_value=loss, lagrangian_penalty=constraint
        )
        if converged:
            break

        step_counter += 1

    assert constraint < 1e-3
    assert torch.isclose(x, torch.tensor(3.0), atol=0.1)
```


you can also utilize pytorch lightning callback to automatically calculate the loss terms
```python
from gradient_utils.optim.schedulers import AugLagLRConfig, AugLagLR, AugLagLossCalculator, AugLagLRCallback

class MyModel(LightningModule):
   def __init__(self):
      super().__init__()
      self.model = ...
      self.training_config = AugLagLRConfig()
      self.lr_scheduler = AugLagLR(config=self.training_config)
      self.loss_calc = AugLagLossCalculator(init_alpha=self.training_config.init_alpha,
                              init_rho=self.training_config.init_rho)
      self.disabled_epochs = range(disabled_epochs)

   def configure_callbacks(self):
      """Create a callback for the auglag callback."""
      return [AuglagLRCallback(self.lr_scheduler, log_auglag=True, disabled_epochs=self.disabled_epochs)]
   
   def training_step(self, batch, batch_idx):
      # loss term should include: loss_terms['loss'], loss_terms['constraint']
      loss_terms = self.compute_loss(batch)
      return loss_terms

```

