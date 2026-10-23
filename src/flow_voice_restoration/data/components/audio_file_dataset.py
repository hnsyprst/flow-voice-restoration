import logging
import math
from itertools import chain
from os import R_OK, access
from pathlib import Path

import numpy as np
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from rich.console import Console
from rich.table import Table
from torchaudio.transforms import Resample

from flow_voice_restoration.utils.audio import audioread

log = logging.getLogger(__name__)


class AudioFileDataset(torch.utils.data.Dataset):
    """
    Basic audio file dataset.
    Each item in the dataset is read from a given directory.
    Pads (with near-zero noise) or crops the audio file to the given duration.
    """

    def __init__(
        self,
        data_dir: str | Path,
        supported_file_extensions: list[str],
        audio_duration: int,
        sample_rate: int,
        num_channels: int,
        strict_validation: bool = True,
    ):
        """
        Initialise the dataset.

        :param data_dir: Directory containing audio files.
        :type data_dir: Union[str, Path]
        :param supported_file_extensions: Audio file extensions to read.
            Do not include the dot (e.g., `["wav", "mp3"]`).
        :type supported_file_extensions: List[str]
        :param audio_duration: Duration (in seconds) of each audio file to return in `__getitem__`.
            Will pad (with near-zero noise) or crop each audio file to fit this duration.
        :type audio_duration: int
        :param sample_rate: Sample rate of the audio. Files not matching this sample rate will be resampled on the fly.
        :type sample_rate: int
        :param num_channels: Number of channels in each audio item.
            Audio files with fewer channels than `num_channels` will have channels added by copying,
            audio files with greater channels than `num_channels` will have channels removed by averaging.
        :type num_channels: int
        :param strict_validation: If True, will raise an exception if any audio files fail validaton.
            Otherwise, will simply exclude the failing file(s) from the dataset.
            Disable this with care; if you fix failing files later, you will break reproducibility for your run.
        :type strict_validation: bool
        """

        super().__init__()

        log.info("Setting up dataset.")

        # Input validation
        data_dir: Path = Path(data_dir)
        if not data_dir.exists(follow_symlinks=True):
            raise ValueError(f"`data_dir` not found. Got {data_dir=}")
        if not data_dir.is_dir():
            raise ValueError(f"`data_dir` must be a directory. Got {type(data_dir)=}")
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.num_channels = num_channels

        # torchaudio Resampler objects can reuse the same kernel
        # if resampling multiple waveforms using the same parameters.
        # We'll cache any Resampler objects we create while iterating over the dataset to speed up resampling
        self.resampler_cache: dict[int, Resample] = dict()

        # Suppress fancy console status output if not rank zero to avoid glitchy output
        console = Console(quiet=rank_zero_only.rank != 0)  # ty:ignore[unresolved-attribute]

        # Recursively glob dataset file paths
        with console.status(f"Reading dataset from {data_dir}...", spinner="earth"):
            # This is a `set` so we can remove erroneous files in O(1). We'll convert it to a list later
            paths: set[Path] = set(
                chain(*[Path(data_dir).rglob(f"*.{extension}") for extension in supported_file_extensions])
            )
        if not paths:
            raise ValueError(
                f"Did not find any files in `data_dir` ({data_dir=}) with extension in {supported_file_extensions}"
            )

        # Check files in dataset are readable
        with console.status("Checking all files for read issues...", spinner="earth"):
            files_errors = dict()
            for path in paths:
                if not path.is_file():
                    files_errors[path] = "Not a file."
                elif not access(path, R_OK):
                    files_errors[path] = "Did not have permission to read file."
        if files_errors:
            table = Table(
                "Path",
                "Error",
                row_styles=["dim", ""],
                title=f"{len(files_errors)} files in the dataset failed read validation:",
                title_justify="left",
                title_style="bold red",
            )
            for path, error in files_errors.items():
                table.add_row(str(path), error)
            # Render via the console: passing the Table to stdlib logging just prints its repr
            console.print(table)
            if strict_validation:
                raise RuntimeError(
                    f"Found {len(files_errors)} files in the dataset failing read validation. "
                    "Since `strict_validation` is True, this is an error. "
                    "See above logs for list of erroneous files."
                )
            else:
                log.info(
                    f"Found {len(files_errors)} files in the dataset failing read validation. "
                    "Since `strict_validation` is False, will skip these files."
                )
                paths -= set(files_errors.keys())

        # Sort so the index -> file mapping is stable across runs (`paths` is a set, whose order isn't)
        self.paths: list[Path] = sorted(paths)
        log.info(f"Finished initialising dataset. Dataset contains {len(self.paths)} files.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(
        self,
        index: int,
    ):
        path = self.paths[index]
        audio, sample_rate = audioread(
            path,
            self.audio_duration,
            read_offset_fraction=np.random.uniform(),
        )
        if sample_rate != self.sample_rate:
            # Create and cache a new resampler for this sample rate if one doesn't already exist.
            # This results in significant speedups when resampling multiple waveforms using the same parameters
            if not self.resampler_cache.get(sample_rate):
                self.resampler_cache[sample_rate] = Resample(sample_rate, self.sample_rate, dtype=audio.dtype)
            audio = self.resampler_cache[sample_rate](audio)
        if audio.shape[0] != self.num_channels:
            if audio.shape[0] < self.num_channels:
                # Upmix by repeating channels cyclically (e.g. mono → stereo)
                repeats = math.ceil(self.num_channels / audio.shape[0])
                audio = audio.repeat((repeats, 1))[: self.num_channels]
            else:
                # Downmix by averaging interleaved channel groups
                # e.g. 6ch → 2ch: avg([0,2,4]), avg([1,3,5])
                audio = torch.stack([audio[i :: self.num_channels].mean(0) for i in range(self.num_channels)])
        return audio
