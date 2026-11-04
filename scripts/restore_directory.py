import argparse
import math
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile
import soxr
import torch
from rich.console import Console

from flow_voice_restoration.inference.restoration import count_windows, restore
from flow_voice_restoration.inference.utils import load_config, load_net
from flow_voice_restoration.utils.rich_logging import TensorProgress

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff"}


def main(
    experiment: str,
    checkpoint: Path,
    input: Path,
    output: Path,
    window_seconds: float,
    overlap_seconds: float,
    batch_size: int,
    device: str,
    num_sampling_steps: int | None,
    guidance_scale: float | None,
    guidance_target: Literal["all", "ssl"],
    temperature: float,
    seed: int | None,
    no_animation: bool,
):
    """
    Restore every audio file under a directory.

    :param experiment: Experiment config the checkpoint was trained with.
    :type experiment: str
    :param checkpoint: Checkpoint to restore with.
    :type checkpoint: Path
    :param input: Directory of audio files, searched recursively.
    :type input: Path
    :param output: Directory to write the restored wavs to.
    :type output: Path
    :param window_seconds: Window length in seconds, snapped to the encoder's frame hop.
    :type window_seconds: float
    :param overlap_seconds: Overlap between windows in seconds, snapped to the encoder's frame hop.
        At most half of `window_seconds`.
    :type overlap_seconds: float
    :param batch_size: How many windows to restore per forward pass.
    :type batch_size: int
    :param device: Device to run the model on.
    :type device: str
    :param num_sampling_steps: Euler steps, or None for the net's own.
    :type num_sampling_steps: int | None
    :param guidance_scale: CFG scale, or None for the net's own.
    :type guidance_scale: float | None
    :param guidance_target: What CFG drops for its weak branch: all conditioning,
        or the SSL content stream alone (for models trained with `ssl_dropout_probability`).
    :type guidance_target: Literal["all", "ssl"]
    :param temperature: Gaussian temperature, the initial noise's variance.
    :type temperature: float
    :param seed: Seed for the sampler, set per file, or None to leave it unseeded.
    :type seed: int | None
    :param no_animation: Don't animate the sampler.
    :type no_animation: bool
    """

    console = Console()
    animate = not no_animation and console.is_terminal

    console.print(f"Experiment: {experiment}")
    console.print(f"Checkpoint: {checkpoint}")

    net = load_net(load_config(experiment), checkpoint)
    net = net.to(device).eval()
    sample_rate = net.sample_rate
    frame_hop = round(sample_rate / net.encoder_frame_rate)
    console.print(f"  {sample_rate} Hz, on {device}")

    # Snap both lengths to the frame hop, so every window starts on the latent grid
    window = max(1, round(window_seconds * sample_rate / frame_hop)) * frame_hop
    overlap = round(overlap_seconds * sample_rate / frame_hop) * frame_hop
    if 2 * overlap > window:
        raise SystemExit(f"--overlap ({overlap_seconds} s) must be at most half of --window ({window_seconds} s).")
    console.print(f"  windows of {window / sample_rate:.2f} s, overlapping by {overlap / sample_rate:.2f} s")

    steps = num_sampling_steps or net.num_sampling_steps
    guidance = guidance_scale if guidance_scale is not None else net.guidance_scale
    if temperature <= 0.0:
        raise SystemExit(f"--temperature must be positive, got {temperature}.")
    if guidance_target == "ssl":
        if net.ssl_conditioner is None:
            raise SystemExit("--guidance-target ssl needs a model with an SSL conditioner.")
        if guidance == 1.0:
            raise SystemExit("--guidance-target ssl does nothing at a guidance scale of 1.0; pass --guidance-scale.")
    sampling_kwargs = {
        "num_sampling_steps": steps,
        "guidance_scale": guidance,
        "temperature": temperature,
        "guidance_target": guidance_target,
    }
    target = " (SSL stream only)" if guidance_target == "ssl" else ""
    console.print(f"  {steps} Euler steps, CFG scale {guidance}{target}, Gaussian temperature {temperature}")

    paths = sorted(p for p in input.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No audio files ({', '.join(sorted(AUDIO_EXTENSIONS))}) under {input}.")
    console.print(f"Restoring {len(paths)} file(s) to {output}")

    for path in paths:
        audio, file_sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if file_sample_rate != sample_rate:
            audio = soxr.resample(audio, file_sample_rate, sample_rate)
        if np.abs(audio).max() < 1e-4:
            console.print(f"  {path.relative_to(input)}: (near-)silent, skipped")
            continue
        audio = torch.from_numpy(audio).float()

        if seed is not None:
            torch.manual_seed(seed)
        if animate:
            num_batches = math.ceil(count_windows(audio.shape[-1], window, overlap) / batch_size)
            with TensorProgress(
                num_batches * steps,
                ramp="blocks",
                color="viridis",
                title=str(path.relative_to(input)),
                console=console,
                transient=True,
                height=20,
            ) as view:
                restored = restore(
                    net,
                    audio,
                    window,
                    overlap,
                    batch_size,
                    device,
                    sampling_kwargs,
                    callback=lambda b, d: view.update(d["denoised"].transpose(1, 2), b * steps + d["i"], d["t"]),
                )
        else:
            restored = restore(net, audio, window, overlap, batch_size, device, sampling_kwargs)

        destination = (output / path.relative_to(input)).with_suffix(".wav")
        destination.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(destination, restored.clamp(-1.0, 1.0).numpy(), sample_rate, subtype="PCM_24")
        console.print(f"  {path.relative_to(input)}: {audio.shape[-1] / sample_rate:.1f}s")

    console.print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment config the checkpoint was trained with",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint to restore with",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory of audio files, searched recursively",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write the restored wavs to",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=10.0,
        help="Window length in seconds (default: the 10 s training crop)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=2.0,
        help="Overlap between windows in seconds",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Windows restored per forward pass",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=None,
        help="Euler steps (default: the net's own)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="CFG scale (default: the net's own)",
    )
    parser.add_argument(
        "--guidance-target",
        choices=["all", "ssl"],
        default="all",
        help="What CFG drops for its weak branch -- all conditioning (default), or the SSL content stream "
        "alone, for models trained with ssl_dropout_probability",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Gaussian temperature, the initial noise's variance (default: 1.0, plain sampling; the CALM "
        "paper uses 0.7-0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the sampler, per file, for a reproducible render",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Don't animate the sampler (it is never animated outside a terminal)",
    )
    args = parser.parse_args()

    main(
        args.experiment,
        args.checkpoint,
        args.input,
        args.output,
        args.window,
        args.overlap,
        args.batch_size,
        args.device,
        args.num_sampling_steps,
        args.guidance_scale,
        args.guidance_target,
        args.temperature,
        args.seed,
        args.no_animation,
    )
