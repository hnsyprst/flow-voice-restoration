from dataclasses import dataclass
from pathlib import Path

import hydra.utils
import torch
from vae.models.components.same import Same


@dataclass(frozen=True)
class ExportedAutoencoder:
    """
    A stage 1 SAME autoencoder together with the sample rate it was trained at, which `Same` does
    not record itself.
    """

    autoencoder: Same
    sample_rate: int


def load_exported_autoencoder(artifact_path: str | Path) -> ExportedAutoencoder:
    """
    Load a SAME autoencoder from an artifact exported by the vae repo's `scripts/export_model.py`.

    The autoencoder is built from the artifact's own config and its weights then loaded over it.
    Building it draws its initial weights from the global RNG, so it must happen at the same point
    relative to every other module's construction for those modules to initialise the same way.

    :param artifact_path: Path to the exported artifact (.pt).
    :type artifact_path: str | Path

    :returns: The autoencoder and its sample rate.
    :rtype: ExportedAutoencoder
    """

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    autoencoder = hydra.utils.instantiate(artifact["net_config"])
    autoencoder.load_state_dict(artifact["model_state_dict"], strict=True)
    return ExportedAutoencoder(autoencoder, int(artifact["sample_rate"]))
