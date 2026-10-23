import logging
from os import R_OK, access
from pathlib import Path

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from rich.console import Console
from rich.table import Table

from flow_voice_restoration.utils.audio import audioread

log = logging.getLogger(__name__)


class PrecomputedRestorationDataset(torch.utils.data.Dataset):
    """
    Dataset of precomputed (degraded audio, clean audio, clean latents) triplets.
    """

    def __init__(
        self,
        data_dir: str | Path,
        strict_validation: bool = True,
        content_feature_suffix: str | None = None,
    ):
        """
        Initialise the dataset.

        :param data_dir: Directory of triplets, as written by `scripts/precompute_dataset.py`.
        :type data_dir: Union[str, Path]
        :param strict_validation: If True, raises if any latents file has no readable matching clean/noisy audio files.
            Otherwise, those items are excluded from the dataset.
            Disable this with care; if you fix failing files later, you will break reproducibility for your run.
        :type strict_validation: bool
        :param content_feature_suffix: Suffix of the SSL content feature tensor to return alongside each triplet,
            e.g. `".contentvec.pt"`. `None` returns triplets, as before this existed.
        :type content_feature_suffix: str | None
        """

        super().__init__()

        log.info("Setting up dataset.")

        data_dir = Path(data_dir)
        if not data_dir.exists(follow_symlinks=True):
            raise ValueError(f"`data_dir` not found. Got {data_dir=}")
        if not data_dir.is_dir():
            raise ValueError(f"`data_dir` must be a directory. Got {data_dir=}")

        # Suppress fancy console status output if not rank zero to avoid glitchy output
        console = Console(quiet=rank_zero_only.rank != 0)  # ty:ignore[unresolved-attribute]

        # Discover latents tensors; each is paired with its audio files by shared stem
        with console.status(f"Reading dataset from {data_dir}...", spinner="earth"):
            latents_paths = sorted(data_dir.rglob("*.latents.pt"))
        if not latents_paths:
            raise ValueError(f"Found no `*.latents.pt` files in `data_dir` ({data_dir=}).")

        # Check each latents file has readable matching clean and noisy audio files,
        # and content features when those were asked for
        with console.status("Checking all files for read issues...", spinner="earth"):
            items: list[tuple[Path, Path, Path, Path | None]] = []
            files_errors: dict[Path, str] = dict()
            for latents_path in latents_paths:
                stem = latents_path.name.removesuffix(".latents.pt")
                clean_path = latents_path.parent / f"{stem}.clean.flac"
                noisy_path = latents_path.parent / f"{stem}.noisy.flac"
                features_path = (
                    None if content_feature_suffix is None else latents_path.parent / f"{stem}{content_feature_suffix}"
                )
                if not access(latents_path, R_OK):
                    files_errors[latents_path] = "Did not have permission to read latents file."
                elif not (clean_path.is_file() and access(clean_path, R_OK)):
                    files_errors[latents_path] = f"No readable matching clean audio file ({clean_path.name})."
                elif not (noisy_path.is_file() and access(noisy_path, R_OK)):
                    files_errors[latents_path] = f"No readable matching noisy audio file ({noisy_path.name})."
                elif features_path is not None and not (features_path.is_file() and access(features_path, R_OK)):
                    files_errors[latents_path] = f"No readable matching content feature file ({features_path.name})."
                else:
                    items.append((noisy_path, clean_path, latents_path, features_path))
        if files_errors:
            table = Table(
                "latents file",
                "Error",
                row_styles=["dim", ""],
                title=f"{len(files_errors)} latents files in the dataset failed read validation:",
                title_justify="left",
                title_style="bold red",
            )
            for path, error in files_errors.items():
                table.add_row(str(path), error)
            # Render via the console: passing the Table to stdlib logging just prints its repr
            console.print(table)
            if strict_validation:
                raise RuntimeError(
                    f"Found {len(files_errors)} latents files in the dataset failing read validation. "
                    "Since `strict_validation` is True, this is an error. "
                    "See above logs for list of erroneous files."
                )
            else:
                log.info(
                    f"Found {len(files_errors)} latents files in the dataset failing read validation. "
                    "Since `strict_validation` is False, will skip these files."
                )

        self.items = items
        log.info(f"Finished initialising dataset. Dataset contains {len(self.items)} items.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, ...]:
        """
        Read one item.

        :param index: Index of the item to read.
        :type index: int

        :returns: `(noisy_audio, clean_audio, clean_latents)`,
            followed by the clean audio's SSL content features when a `content_feature_suffix` was given.
        :rtype: tuple[torch.Tensor, ...]
        """

        noisy_path, clean_path, latents_path, features_path = self.items[index]
        noisy_audio, _ = audioread(noisy_path)
        clean_audio, _ = audioread(clean_path)
        clean_latents = torch.load(latents_path)
        if features_path is None:
            return noisy_audio, clean_audio, clean_latents
        content_features = torch.load(features_path, weights_only=True)
        return noisy_audio, clean_audio, clean_latents, content_features
