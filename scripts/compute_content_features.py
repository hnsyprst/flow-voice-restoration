import argparse
import os
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import track
from torch.utils.data import DataLoader, Dataset, Subset
from torchaudio.functional import resample
from transformers import AutoModel

from flow_voice_restoration.utils.audio import audioread

#: ContentVec inherits HuBERT's frontend: a 400-sample window every 320 samples at 16 kHz.
MODEL_SAMPLE_RATE = 16000
WINDOW_SAMPLES = 400
HOP_SAMPLES = 320

#: What `precompute_dataset.py` writes, and what this script writes beside it.
CLEAN_SUFFIX = ".clean.flac"
FEATURE_SUFFIX = ".contentvec.pt"

DTYPES = {"float16": torch.float16, "float32": torch.float32}

console = Console()


class CleanClipDataset(Dataset):
    """
    The clean clips of a precomputed set, resampled to the model's rate.
    """

    def __init__(
        self,
        paths: list[Path],
    ):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:
        audio, sample_rate = audioread(self.paths[index])
        audio = audio.mean(dim=0)
        if sample_rate != MODEL_SAMPLE_RATE:
            audio = resample(audio, sample_rate, MODEL_SAMPLE_RATE)
        return audio, index


def num_frames(
    num_samples: int,
) -> int:
    return 1 + max(0, num_samples - WINDOW_SAMPLES) // HOP_SAMPLES


def main(
    args: argparse.Namespace,
) -> None:
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    model = AutoModel.from_pretrained(args.model_name).to(device).eval()
    num_layers = model.config.num_hidden_layers
    if args.layer is not None and not 0 <= args.layer <= num_layers:
        raise ValueError(f"`--layer` must be in [0, {num_layers}] for {args.model_name}. Got {args.layer}.")

    paths = sorted(path for path in args.data_root.rglob(f"*{CLEAN_SUFFIX}") if path.is_file())
    if not paths:
        raise ValueError(f"Found no `*{CLEAN_SUFFIX}` files under {args.data_root}")

    def output_path(path: Path) -> Path:
        return path.with_name(path.name.removesuffix(CLEAN_SUFFIX) + FEATURE_SUFFIX)

    if args.skip_existing:
        remaining = [path for path in paths if not output_path(path).is_file()]
        if is_main:
            console.log(f"{len(paths) - len(remaining)} of {len(paths)} already computed; skipping those.")
        paths = remaining

    dataset = CleanClipDataset(paths)
    shard = (len(dataset) + world_size - 1) // world_size
    subset = Subset(dataset, range(rank * shard, min((rank + 1) * shard, len(dataset))))

    if is_main:
        layer = "last" if args.layer is None else args.layer
        console.log(
            f"{args.model_name} layer {layer} over {len(dataset)} clip(s) across {world_size} GPU(s), "
            f"batches of {args.batch_size}, saved as {args.dtype}."
        )
        console.log(f"Writing `*{FEATURE_SUFFIX}` beside the clips under {args.data_root}")

    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    num_written = 0
    for audio_batch, indices in track(dataloader, disable=not is_main):
        with torch.inference_mode():
            outputs = model(audio_batch.to(device), output_hidden_states=args.layer is not None)
        features_batch = outputs.last_hidden_state if args.layer is None else outputs.hidden_states[args.layer]
        # Every clip in a precomputed set is the same length, so one trim serves the whole batch
        clip_frames = num_frames(audio_batch.shape[-1])

        for features, index in zip(features_batch, indices.tolist(), strict=True):
            features = features[:clip_frames].to(DTYPES[args.dtype]).clone().cpu()
            torch.save(features, output_path(dataset.paths[index]))
            num_written += 1

    console.log(f"Done. Rank {rank} wrote {num_written} of its {len(subset)} shard clip(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "data_root",
        type=Path,
        help="A set written by scripts/precompute_dataset.py",
    )
    parser.add_argument(
        "--model-name",
        default="lengyue233/content-vec-best",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=6,
        help="Hidden-state index to save; -1 for the last layer",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=sorted(DTYPES),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip clips whose features already exist, so an interrupted run can be resumed",
    )
    args = parser.parse_args()
    if args.layer == -1:
        args.layer = None

    main(args)
