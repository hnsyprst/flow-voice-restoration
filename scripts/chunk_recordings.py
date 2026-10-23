import argparse
import re
from pathlib import Path

import numpy as np
import soundfile
from rich.console import Console
from rich.progress import track


def chunk(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    min_seconds: float,
) -> list[np.ndarray]:
    """
    Cut one recording into consecutive chunks.

    :param audio: Mono audio, shape `(num_samples,)`.
    :type audio: np.ndarray
    :param sample_rate: Sample rate of `audio` in Hz.
    :type sample_rate: int
    :param chunk_seconds: Length of every full chunk.
    :type chunk_seconds: float
    :param min_seconds: Shortest piece kept: a recording shorter than a chunk, or the remainder after
        the last full one, is kept only if it is at least this long.
    :type min_seconds: float

    :returns: The chunks, in order.
    :rtype: list[np.ndarray]
    """

    size = round(chunk_seconds * sample_rate)
    pieces = [audio[start : start + size] for start in range(0, len(audio), size)]
    return [piece for piece in pieces if len(piece) == size or len(piece) >= min_seconds * sample_rate]


def main(
    input_root: Path,
    output_root: Path,
    supported_file_extensions: list[str],
    chunk_seconds: float,
    min_seconds: float,
    exclude: list[str],
):
    console = Console()
    patterns = [re.compile(pattern) for pattern in exclude]
    paths = sorted(path for extension in supported_file_extensions for path in input_root.rglob(f"*.{extension}"))
    kept = [path for path in paths if not any(pattern.search(path.name) for pattern in patterns)]
    console.log(f"{len(paths)} recordings, {len(paths) - len(kept)} excluded by name.")

    num_chunks, seconds, dropped_seconds = 0, 0.0, 0.0
    for path in track(kept, description="Chunking..."):
        audio, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(1)
        pieces = chunk(audio, sample_rate, chunk_seconds, min_seconds)
        target = output_root / path.relative_to(input_root).with_suffix("")
        target.parent.mkdir(parents=True, exist_ok=True)
        for index, piece in enumerate(pieces):
            soundfile.write(target.with_name(f"{target.name}_{index:03d}.flac"), piece, sample_rate, subtype="PCM_24")
        kept_samples = sum(map(len, pieces))
        seconds += kept_samples / sample_rate
        dropped_seconds += (len(audio) - kept_samples) / sample_rate
        num_chunks += len(pieces)
    console.log(
        f"Wrote {num_chunks} chunks, {seconds / 3600:.1f} h, to {output_root}; "
        f"dropped {dropped_seconds / 3600:.2f} h of short remainders."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_root",
        type=Path,
        help="Directory tree of recordings",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Output directory (mirrors the input structure)",
    )
    parser.add_argument(
        "-e",
        "--supported-file-extensions",
        nargs="+",
        default=["wav", "flac"],
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=10.5,
        help="Length of each chunk",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=5.0,
        help="Shortest recording or final remainder kept",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="REGEX",
        help="Skip recordings whose file name matches this regex (repeatable)",
    )
    args = parser.parse_args()

    main(
        args.input_root,
        args.output_root,
        args.supported_file_extensions,
        args.chunk_seconds,
        args.min_seconds,
        args.exclude,
    )
