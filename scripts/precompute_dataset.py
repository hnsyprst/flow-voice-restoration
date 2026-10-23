import argparse
import os
import random
from pathlib import Path

import numpy as np
import soundfile
import torch
import torch.nn.functional as F  # noqa: N812
from rich.console import Console
from rich.progress import track
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from flow_voice_restoration.data.components.audio_file_dataset import AudioFileDataset
from flow_voice_restoration.data.components.degradations import (
    RandomAudioSampler,
    degrade,
    degrade_silence,
    lowcut,
    normalize,
)
from flow_voice_restoration.utils.autoencoder import load_exported_autoencoder


class DegradationDataset(Dataset):
    """
    Wraps a clean `AudioFileDataset` and applies the degradation chain.
    """

    def __init__(
        self,
        clean_dataset: AudioFileDataset,
        noise_ds: RandomAudioSampler,
        degradation_prob: float,
        seed: int,
    ):
        self.clean_dataset = clean_dataset
        self.noise_ds = noise_ds
        self.degradation_prob = degradation_prob
        self.seed = seed

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(
        self,
        index: int,
    ) -> dict | None:
        random.seed(self.seed * 1_000_003 + index)
        np.random.seed((self.seed * 1_000_003 + index) % 2**32)

        audio = self.clean_dataset[index]
        if audio.abs().max() < 1e-4:
            # (Near-)silent clip, normalizing would just amplify noise floor, so skip it
            return None
        sample_rate = self.clean_dataset.sample_rate
        clean = normalize(lowcut(audio, sample_rate))
        clean, noisy = degrade(clean, sample_rate, self.noise_ds, prob=self.degradation_prob)
        # Peak-limit the degraded audio (e.g. after low-SNR noise addition) so it survives the FLAC write
        peak = noisy.abs().max()
        if peak > 0.99:
            noisy = noisy * (0.99 / peak)
        return {"index": index, "clean": clean, "noisy": noisy}


class SilentTargetDataset(Dataset):
    """
    Dataset of noise-only clips with silent targets.
    """

    def __init__(
        self,
        num_items: int,
        num_samples: int,
        sample_rate: int,
        noise_ds: RandomAudioSampler,
        degradation_prob: float,
        level_dbfs_range: tuple[float, float],
        seed: int,
    ):
        self.num_items = num_items
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.noise_ds = noise_ds
        self.degradation_prob = degradation_prob
        self.level_dbfs_range = level_dbfs_range
        self.seed = seed

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        random.seed(self.seed * 1_000_003 + index)
        np.random.seed((self.seed * 1_000_003 + index) % 2**32)

        clean, noisy = degrade_silence(
            self.num_samples,
            self.sample_rate,
            self.noise_ds,
            prob=self.degradation_prob,
            level_dbfs_range=self.level_dbfs_range,
        )
        peak = noisy.abs().max()
        if peak > 0.99:
            noisy = noisy * (0.99 / peak)
        return {"index": index, "clean": clean, "noisy": noisy}


def collate_skip_none(items: list[dict | None]) -> dict | None:
    items = [item for item in items if item is not None]
    return default_collate(items) if items else None


def main(
    input_root: Path,
    output_root: Path,
    autoencoder_artifact: Path,
    noise_dirs: list[Path],
    supported_file_extensions: list[str],
    audio_duration: int,
    sample_rate: int | None,
    degradation_prob: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    limit: int | None,
    overwrite: bool,
    silent_targets: int | None = None,
    silent_level_range: tuple[float, float] = (-40.0, -10.0),
):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    console = Console()
    if is_main:
        console.log("Precomputing the voice-restoration dataset.")
        console.log("Setting up the frozen autoencoder.")

    exported = load_exported_autoencoder(autoencoder_artifact)
    autoencoder = exported.autoencoder.to(device)
    autoencoder.eval()
    autoencoder.requires_grad_(False)

    if sample_rate is None:
        sample_rate = exported.sample_rate
        if is_main:
            console.log(f"Reading audio at the artifact's own sample rate ({sample_rate} Hz).")

    if is_main:
        console.log("Setting up dataset.")

    noise_ds = RandomAudioSampler(noise_dirs, duration=audio_duration)
    if silent_targets is None:
        clean_dataset = AudioFileDataset(
            data_dir=input_root,
            supported_file_extensions=supported_file_extensions,
            audio_duration=audio_duration,
            sample_rate=sample_rate,
            num_channels=1,
            strict_validation=False,
        )
        dataset = DegradationDataset(clean_dataset, noise_ds, degradation_prob=degradation_prob, seed=seed)

        def output_path(index: int) -> Path:
            return output_root / clean_dataset.paths[index].relative_to(input_root)

        source_description = f"{len(clean_dataset)} clean files"
    else:
        dataset = SilentTargetDataset(
            silent_targets,
            num_samples=int(audio_duration) * sample_rate,
            sample_rate=sample_rate,
            noise_ds=noise_ds,
            degradation_prob=degradation_prob,
            level_dbfs_range=silent_level_range,
            seed=seed,
        )

        def output_path(index: int) -> Path:
            return output_root / f"silent_{index:06d}"

        source_description = f"{silent_targets} silent-target clips"

    def is_done(index: int) -> bool:
        path = output_path(index)
        return all(path.with_suffix(suffix).exists() for suffix in (".clean.flac", ".noisy.flac", ".latents.pt"))

    num_items = len(dataset) if limit is None else min(limit, len(dataset))
    remaining = [index for index in range(num_items) if overwrite or not is_done(index)]

    chunk_size = (len(remaining) + world_size - 1) // world_size
    chunk = remaining[rank * chunk_size : (rank + 1) * chunk_size]
    dataloader = DataLoader(
        Subset(dataset, chunk),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
        shuffle=False,
    )

    if is_main:
        console.log(
            f"{source_description}; {len(remaining)}/{num_items} item(s) to process "
            f"across {world_size} process(es) on {device.type} in batches of {batch_size}."
        )
        console.log(f"Will write clean/noisy audio and latents to {output_root}.")

    num_written = 0
    for batch in track(dataloader, description="Precomputing...", disable=not is_main):
        if batch is None:
            continue
        clean = batch["clean"].to(device)
        latents, _ = autoencoder.encode(F.pad(clean, (0, -clean.shape[-1] % autoencoder.patch_size)))
        latents = latents.cpu()
        for i, index in enumerate(batch["index"].tolist()):
            path = output_path(index)
            path.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(
                path.with_suffix(".clean.flac"), batch["clean"][i].squeeze(0).numpy(), sample_rate, subtype="PCM_24"
            )
            soundfile.write(
                path.with_suffix(".noisy.flac"), batch["noisy"][i].squeeze(0).numpy(), sample_rate, subtype="PCM_24"
            )
            torch.save(latents[i].clone(), path.with_suffix(".latents.pt"))
            num_written += 1

    console.log(
        f"Done. Rank {rank} wrote {num_written} of its {len(chunk)} shard item(s) "
        f"({len(chunk) - num_written} skipped as near-silent); "
        f"{len(remaining)} were shared across {world_size} process(es)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_root",
        type=Path,
        nargs="?",
        help="Directory tree of clean audio files (omit with --silent-targets)",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Output directory (mirrors the input structure)",
    )
    parser.add_argument(
        "autoencoder_artifact",
        type=Path,
        help="SAME autoencoder artifact exported by vae's scripts/export_model.py",
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        action="append",
        required=True,
        dest="noise_dirs",
        help="Directory tree(s) of noise recordings (repeatable)",
    )
    parser.add_argument(
        "-e",
        "--supported-file-extensions",
        type=str,
        nargs="+",
        required=False,
        default=["wav", "flac"],
    )
    parser.add_argument(
        "-d",
        "--audio-duration",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "-s",
        "--sample-rate",
        type=int,
        required=False,
        default=None,
        help="Sample rate to read the audio at. Defaults to the autoencoder artifact's own sample rate",
    )
    parser.add_argument(
        "-p",
        "--degradation-prob",
        type=float,
        required=False,
        default=0.5,
        help="Probability of applying each degradation",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=False,
        default=16,
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        required=False,
        default=16,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=False,
        default=None,
        help="Only process the first N items (for smoke tests)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute items whose outputs already exist",
    )
    parser.add_argument(
        "--silent-targets",
        type=int,
        default=None,
        metavar="N",
        help="Write N silent-target clips (noise in, silence out) instead of degrading input_root",
    )
    parser.add_argument(
        "--silent-level-range",
        type=float,
        nargs=2,
        default=(-40.0, -10.0),
        metavar=("LOW", "HIGH"),
        help="With --silent-targets: range of the noise's RMS level, in dBFS",
    )
    args = parser.parse_args()
    if (args.input_root is None) == (args.silent_targets is None):
        parser.error("Give exactly one of input_root and --silent-targets.")

    main(
        args.input_root,
        args.output_root,
        args.autoencoder_artifact,
        args.noise_dirs,
        args.supported_file_extensions,
        args.audio_duration,
        args.sample_rate,
        args.degradation_prob,
        args.batch_size,
        args.num_workers,
        args.seed,
        args.limit,
        args.overwrite,
        args.silent_targets,
        tuple(args.silent_level_range),
    )
