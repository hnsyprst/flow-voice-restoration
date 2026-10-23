import logging
import os
import sys
from pathlib import Path

import hydra
import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from flow_voice_restoration.utils.resolvers import register_resolvers

# Log full error tracebacks
os.environ["HYDRA_FULL_ERROR"] = "1"
# Set PYTHONPATH based on `.project-root` file location
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# Set up logger, capture warnings and PyTorch Lightning logs
logging.captureWarnings(True)
log = logging.getLogger(__name__)
for name in ["lightning", "lightning.pytorch", "transformers"]:
    lg = logging.getLogger(name)
    lg.handlers.clear()
    lg.propagate = True
# Register all omegaconf resolvers
register_resolvers()


def handle_exception(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)  # let Ctrl+C behave normally
        return
    log.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))


sys.excepthook = handle_exception


@hydra.main(version_base="1.3", config_path="../../configs", config_name="default.yaml")
def main(cfg: DictConfig):
    log.info("Starting a new training job.")
    log.info(f":card_index: {'[bold green]Job Name:[/]':<35}{cfg.run_name}")
    log.info(f":page_facing_up: {'[bold green]Description:[/]':<35}{cfg.description}")
    log.info(
        f":file_folder: {'[bold green]Output directory:[/]':<35}"
        f"./{Path(HydraConfig.get().run.dir).relative_to(rootutils.find_root())}"
    )

    if seed := cfg.get("seed"):
        log.info(f"Seed set to {seed}.")
        seed_everything(seed, workers=True, verbose=False)

    if float32_matmul_precision := cfg.get("float32_matmul_precision"):
        log.info(
            f"Using '{float32_matmul_precision}' precision float32 matmuls. "
            "'medium' or 'high' precision will trade off precision for speed. "
            "Configure this via `float32_matmul_precision` in the config."
        )
        torch.set_float32_matmul_precision(float32_matmul_precision)

    # Instantiate datamodule and trainer
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Instantiate callbacks
    callbacks: list[Callback] = []
    if cfg.callbacks is not None:
        for callback_cfg in cfg.callbacks.values():
            if isinstance(callback_cfg, DictConfig) and "_target_" in callback_cfg:
                callbacks.append(hydra.utils.instantiate(callback_cfg))
    else:
        log.warning("No callbacks found in config.")

    # Instantiate loggers
    loggers: list[Logger] = []
    if cfg.loggers is not None:
        for logger_cfg in cfg.loggers.values():
            if isinstance(logger_cfg, DictConfig) and "_target_" in logger_cfg:
                loggers.append(hydra.utils.instantiate(logger_cfg))
    else:
        log.warning("No loggers found in config.")

    # Instantiate trainer and start training
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.get("ckpt_path"),
        weights_only=False,
    )


if __name__ == "__main__":
    main()
