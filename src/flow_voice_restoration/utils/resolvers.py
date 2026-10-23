import os

import hydra
import omegaconf
from omegaconf import OmegaConf

from flow_voice_restoration.utils.mlflow import get_mlflow_tracking_uri_and_authenticate


def register_resolvers():
    OmegaConf.register_new_resolver("hydra_module", lambda: hydra)
    OmegaConf.register_new_resolver("omegaconf_module", lambda: omegaconf)
    OmegaConf.register_new_resolver("mlflow_tracking_uri", get_mlflow_tracking_uri_and_authenticate)
    OmegaConf.register_new_resolver("local_username", os.getlogin)
