"""
CLI entry point for training a trainable KANTauTheta inside LSFNO (Study 2).

Usage (from LS_KAN_FNO root, with venv activated):
  python -m training.train --config configs/training_linear_prototype.yaml
"""

import argparse

from training.trainer import Trainer
from utils.config_loader import load_config


def _parse_args():
    p = argparse.ArgumentParser(description="Train the linear LS-KAN-FNO (Study 2).")
    p.add_argument('--config', type=str, required=True,
                   help="Path to a training_*.yaml config file.")
    p.add_argument('--n_epochs', type=int, default=None,
                   help="Override n_epochs from config.")
    p.add_argument('--lbfgs', action='store_true',
                   help="Run the optional LBFGS refinement phase after fit().")
    return p.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    if args.n_epochs is not None:
        config['n_epochs'] = args.n_epochs

    trainer = Trainer(config)
    trainer.fit()

    if args.lbfgs or config.get('lbfgs_phase', False):
        final_loss = trainer.fit_lbfgs()
        print(f"LBFGS refinement done. final loss = {final_loss:.6e}")


if __name__ == '__main__':
    main()
