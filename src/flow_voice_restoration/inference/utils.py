from pathlib import Path

import rootutils
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from flow_voice_restoration.models.components.voice_restoration import VoiceRestorer


def load_config(
    experiment: str,
    config_dir: Path | str | None = None,
) -> DictConfig:
    """
    Compose an experiment's config as `train.py experiment=<experiment>` would.

    :param experiment: Experiment config name.
    :type experiment: str
    :param config_dir: Path to directory containing the config.
        If this is `None`, defaults to the config directory in the repository root.
    :type config_dir: Path | str | None

    :returns: The composed config.
    :rtype: DictConfig
    """

    if config_dir is None:
        config_dir = rootutils.find_root(__file__, indicator=".project-root") / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name="default", overrides=[f"experiment={experiment}"])


def load_net(
    config: DictConfig,
    checkpoint_path: Path,
) -> VoiceRestorer:
    """
    Instantiate the voice restorer from an experiment's config and load a checkpoint's weights into it.

    :param config: The experiment's composed config.
    :type config: DictConfig
    :param checkpoint_path: Lightning checkpoint holding the `net.*` weights.
    :type checkpoint_path: Path

    :returns: The restorer.
    :rtype: VoiceRestorer
    """

    net = instantiate(config.model.net)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        for prefix in ("net._orig_mod.", "net."):
            if key.startswith(prefix):
                state_dict[key[len(prefix) :]] = value
                break

    net.load_state_dict(state_dict, strict=True)
    return net
