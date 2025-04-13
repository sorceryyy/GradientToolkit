import torch
import pytest
from torch.nn import Parameter
from torch.optim import Adam
from pytorch_lightning import LightningModule, Trainer
from gradient_utils.optim.schedulers.auglag import (
    AugLagLRConfig,
    AugLagLR,
    AugLagLossCalculator,
    AuglagLRCallback,
)


def test_solve_simple_constraint():
    """
    Test that `AugLagLR` can solve a simple constrained optimization problem:
        min x² s.t. x > 3
    The solution should be x ≈ 3
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
    max_iter = 10000
    
    for _ in range(max_iter):
        optimizer.zero_grad()
        loss = x**2
        constraint = torch.max(3 - x, torch.zeros(()))
        auglag_loss_tensor = auglag_loss(loss, constraint)
        auglag_loss_tensor.backward()
        optimizer.step()
        
        converged = scheduler.step(
            optimizer=optimizer,
            loss=auglag_loss,
            loss_value=loss,
            lagrangian_penalty=constraint
        )
        if converged:
            break

    assert constraint < 1e-3
    assert torch.isclose(x, torch.tensor(3.0), atol=0.1)


class DummyModel(LightningModule):
    """Dummy model for testing AugLagLR with PyTorch Lightning."""
    
    def __init__(self):
        super().__init__()
        self.x = Parameter(torch.zeros(()))
        self.training_config = AugLagLRConfig(
            lr_init_dict={"x": 0.1},
            max_outer_steps=1000,
        )
        self.lr_scheduler = AugLagLR(config=self.training_config)
        self.loss_calc = AugLagLossCalculator(
            init_alpha=self.training_config.init_alpha,
            init_rho=self.training_config.init_rho
        )
        self.disabled_epochs = range(0)  # No disabled epochs

    def configure_callbacks(self):
        return [AuglagLRCallback(
            self.lr_scheduler,
            log_auglag=True,
            disabled_epochs=self.disabled_epochs
        )]

    def configure_optimizers(self):
        optimizer = Adam([{"params": self.x, "name": "x", "lr": 0.1}])
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.x**2
        constraint = torch.max(3 - self.x, torch.zeros(()))
        return {
            "loss": loss,
            "constraint": constraint
        }


@pytest.mark.skip(reason="This test requires a proper training environment")
def test_lightning_integration():
    """Test AugLagLR integration with PyTorch Lightning."""
    model = DummyModel()
    trainer = Trainer(
        max_epochs=100,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model)
    
    # Check if optimization succeeded
    assert model.x > 2.9  # x should be close to 3
    constraint = torch.max(3 - model.x, torch.zeros(()))
    assert constraint < 1e-3  # constraint should be satisfied
