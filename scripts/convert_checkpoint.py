import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace

import torch

REFERENCE_PACKAGE = "voice_restoration"
PACKAGE = "flow_voice_restoration"

RENAMED_PREFIXES = {
    "net.encoder.encoder.": "net.encoder.",
    "net.bottleneck.bottleneck_module.": "net.bottleneck.",
    "net.decoder.decoder.": "net.decoder.",
}

REMOVED_HYPERPARAMETERS = [
    "feature_matching_criterion_weight",
    "feature_matching_mode",
    "feature_matching_min_timestep",
    "num_feature_matching_sampling_steps",
]

REMOVED_DATAMODULE_HYPERPARAMETERS = [
    "multiprocessing_context",
]


class RenamingUnpickler(pickle.Unpickler):
    """
    Unpickler that resolves the voice-restoration package's classes and functions to this one's.
    """

    def find_class(
        self,
        module: str,
        name: str,
    ):
        if module == REFERENCE_PACKAGE or module.startswith(f"{REFERENCE_PACKAGE}."):
            module = PACKAGE + module[len(REFERENCE_PACKAGE) :]
        return super().find_class(module, name)


def renamed(key: str) -> str:
    for old, new in RENAMED_PREFIXES.items():
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def convert(checkpoint: dict) -> dict:
    """
    Convert a voice-restoration Lightning checkpoint in place.

    :param checkpoint: The checkpoint, as loaded.
    :type checkpoint: dict

    :returns: The same checkpoint, converted.
    :rtype: dict
    """

    checkpoint["state_dict"] = {renamed(key): value for key, value in checkpoint["state_dict"].items()}
    for name in REMOVED_HYPERPARAMETERS:
        checkpoint.get("hyper_parameters", {}).pop(name, None)
    for name in REMOVED_DATAMODULE_HYPERPARAMETERS:
        checkpoint.get("datamodule_hyper_parameters", {}).pop(name, None)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input",
        type=Path,
        help="Checkpoint written by the voice-restoration repo",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Where to write the converted checkpoint",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"{args.output} already exists.")

    checkpoint = torch.load(
        args.input,
        map_location="cpu",
        weights_only=False,
        pickle_module=SimpleNamespace(__name__="renaming_pickle", Unpickler=RenamingUnpickler, load=pickle.load),
    )
    num_renamed = sum(renamed(key) != key for key in checkpoint["state_dict"])
    checkpoint = convert(checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Renamed {num_renamed} of {len(checkpoint['state_dict'])} weights; wrote {args.output}")


if __name__ == "__main__":
    main()
