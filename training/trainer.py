"""
Trainer for Study 2: make KANTauTheta's control points trainable and confirm
they recover the exact bilinear contraction T:epsilon from FFT-generated data.

Only the KAN side of tau_theta is ever trained here — the Yarotsky MLP
(models/ls_fno.py::YarotskyTauTheta) is analytic by construction and stays
fixed; it's used only as a comparison baseline (training/metrics.py,
evaluation/).

Usage:
    from utils.config_loader import load_config
    from training.trainer import Trainer

    config = load_config("configs/training_linear_prototype.yaml")
    trainer = Trainer(config)
    trainer.fit()
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.micromechanics import DataLoaderFactory
from models.kan_tau_theta import KANTauTheta
from models.ls_fno import LSFNO
from training.losses import combined_loss
from training.metrics import gamma_theta_bound, gamma_theta_empirical


class _LocalLogger:
    """Default logger: append-only JSON-lines file + a print per epoch. No
    account or external service needed — used unless use_wandb is set."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / "history.jsonl"
        self._fh = open(self.path, "a")

    def log(self, row: dict):
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        parts = [f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()]
        print("  ".join(parts))

    def close(self):
        self._fh.close()


class _WandbLogger:
    def __init__(self, config: dict, output_dir: Path):
        import wandb
        self.run = wandb.init(
            project=config.get("wandb_project", "ls_kan_fno"),
            config=config,
            dir=str(output_dir),
        )

    def log(self, row: dict):
        self.run.log(row)

    def close(self):
        self.run.finish()


def _build_logger(config: dict, output_dir: Path):
    if config.get("use_wandb", False):
        try:
            return _WandbLogger(config, output_dir)
        except ImportError:
            print("use_wandb=True but wandb is not installed; falling back to local logger.")
    return _LocalLogger(output_dir)


class Trainer:
    """Builds an LSFNO with a trainable KANTauTheta and runs AdamW + cosine
    annealing, with an optional LBFGS refinement phase afterwards."""

    def __init__(self, config: dict):
        self.config = config
        torch.manual_seed(int(config.get("seed", 42)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loaders = DataLoaderFactory.from_config(config)
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.test_loader = loaders["test"]

        dim = int(config.get("dim", 2))
        n_comp = 3 if dim == 2 else 6
        tau_theta = KANTauTheta(
            R=float(config.get("R", 1.0)),
            shared=bool(config.get("shared", True)),
            trainable=True,
            n_comp=n_comp,
        )
        self.model = LSFNO.from_config(config, tau_theta=tau_theta).to(self.device)

        # Override ctrl initialization if requested (default "exact" = [1,-1,1]).
        # For Study 2, use a wrong starting point to test gradient-descent recovery.
        ctrl_init = config.get("ctrl_init", "exact")
        if ctrl_init != "exact":
            with torch.no_grad():
                ctrl = self.model.tau_theta.ctrl
                if ctrl_init == "random":
                    torch.manual_seed(int(config.get("seed", 42)))
                    ctrl.data = torch.rand_like(ctrl) * 2.0 - 1.0
                elif ctrl_init == "zero":
                    ctrl.data.zero_()
                elif isinstance(ctrl_init, (list, tuple)):
                    ctrl.data = torch.tensor(ctrl_init, dtype=ctrl.dtype, device=ctrl.device)
                else:
                    raise ValueError(f"Unknown ctrl_init: {ctrl_init!r}. "
                                     f"Use 'exact', 'random', 'zero', or a list [c0,c1,c2].")
            print(f"ctrl_init='{ctrl_init}' → ctrl = {self.model.tau_theta.ctrl.detach().cpu().numpy()}")
        self.n_normal = self.model.n_normal
        self.use_checkpointing = bool(config.get("use_checkpointing", False))

        self.lambda_field = float(config.get("lambda_field", 1.0))
        self.lambda_eff = float(config.get("lambda_eff", 0.1))

        self.n_epochs = int(config.get("n_epochs", 100))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.get("lr", 1e-3)),
            # KAN control points directly parametrize phi(x); decaying them
            # toward 0 would bias away from the correct [1,-1,1] answer, so
            # weight decay defaults off (unlike a typical AdamW use case).
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.n_epochs, eta_min=float(config.get("lr_min", 1e-5)),
        )

        self.output_dir = Path(config.get("output_dir", "runs/default"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = int(config.get("checkpoint_every", 10))
        self.val_every = int(config.get("val_every", 1))
        self.gamma_check_every = int(config.get("gamma_theta_check_every", 5))
        self.gamma_bound = gamma_theta_bound(float(config.get("kappa", 10.0)))

        self.logger = _build_logger(config, self.output_dir)
        self.best_val_loss = float("inf")
        self.history = []

    # ── internals ──────────────────────────────────────────────────────────

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _forward_batch(self, batch: dict):
        batch = self._to_device(batch)
        eps_pred = self.model(
            batch["C_field"], batch["eps_bar"], use_checkpointing=self.use_checkpointing
        )
        loss, parts = combined_loss(
            eps_pred, batch["eps_star"], batch["C_field"], batch["eps_bar"], batch["sigma_star"],
            n_normal=self.n_normal, lambda_field=self.lambda_field, lambda_eff=self.lambda_eff,
        )
        return loss, parts, batch

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        totals = {"total": 0.0, "field": 0.0, "eff": 0.0}
        n_batches = 0
        with torch.enable_grad() if train else torch.no_grad():
            for batch in loader:
                if train:
                    self.optimizer.zero_grad()
                loss, parts, _ = self._forward_batch(batch)
                if train:
                    loss.backward()
                    self.optimizer.step()
                totals["total"] += float(loss.detach())
                totals["field"] += float(parts["field"])
                totals["eff"] += float(parts["eff"])
                n_batches += 1
        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def _save_checkpoint(self, epoch: int, tag: str):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": self.best_val_loss,
            "config": self.config,
        }
        torch.save(ckpt, self.output_dir / f"{tag}_checkpoint.pt")

    # ── public API ─────────────────────────────────────────────────────────

    def fit(self) -> list:
        """AdamW + cosine annealing main training loop. Returns the per-epoch
        history (also written to <output_dir>/history.jsonl)."""
        for epoch in range(1, self.n_epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            self.scheduler.step()

            log_row = {
                "epoch": epoch,
                "lr": self.scheduler.get_last_lr()[0],
                **{f"train_{k}": v for k, v in train_metrics.items()},
            }

            if epoch % self.val_every == 0:
                val_metrics = self._run_epoch(self.val_loader, train=False)
                log_row.update({f"val_{k}": v for k, v in val_metrics.items()})

                if epoch % self.gamma_check_every == 0:
                    batch = self._to_device(next(iter(self.val_loader)))
                    gamma = gamma_theta_empirical(self.model, batch)
                    log_row.update({f"gamma_theta_{k}": v for k, v in gamma.items()})
                    log_row["gamma_theta_bound"] = self.gamma_bound

                if val_metrics["total"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total"]
                    self._save_checkpoint(epoch, "best")

            if epoch % self.checkpoint_every == 0 or epoch == self.n_epochs:
                self._save_checkpoint(epoch, "last")

            self.history.append(log_row)
            self.logger.log(log_row)

        self.logger.close()
        return self.history

    def fit_lbfgs(self):
        """Optional refinement phase: a handful of LBFGS steps on one small
        batch, run after fit(). Mainly useful once shared=False (27 params) —
        with shared=True (3 params) Adam alone is normally enough."""
        steps = int(self.config.get("lbfgs_steps", 30))
        batch_size = int(self.config.get("lbfgs_batch_size", 4))

        small_loader = DataLoader(self.train_loader.dataset, batch_size=batch_size, shuffle=True)
        batch = self._to_device(next(iter(small_loader)))

        self.model.train(True)
        optimizer = torch.optim.LBFGS(
            self.model.parameters(), lr=1.0, max_iter=steps, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss, _, _ = self._forward_batch(batch)
            loss.backward()
            return loss

        final_loss = optimizer.step(closure)
        self._save_checkpoint(self.n_epochs, "last_lbfgs")
        return float(final_loss)
