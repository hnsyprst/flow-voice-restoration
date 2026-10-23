import argparse
import math
from pathlib import Path

import numpy as np
import rootutils
import soundfile
import soxr
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from flow_voice_restoration.models.components.voice_restoration import VoiceRestorer

CONFIG_DIR = rootutils.find_root(__file__, indicator=".project-root") / "configs"

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff"}

#: The peak the training clips were normalised to (`degradations.normalize`)
INPUT_PEAK = 0.9


def load_config(
    experiment: str,
) -> DictConfig:
    """
    Compose an experiment's config as `train.py experiment=<experiment>` would.

    :param experiment: Experiment config name.
    :type experiment: str

    :returns: The composed config.
    :rtype: DictConfig
    """

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="default", overrides=[f"experiment={experiment}"])


def load_net(
    config: DictConfig,
    checkpoint_path: Path,
) -> tuple[VoiceRestorer, int | None, int | None]:
    """
    Instantiate the voice restorer from an experiment's config and load a checkpoint's weights into it.

    :param config: The experiment's composed config.
    :type config: DictConfig
    :param checkpoint_path: Lightning checkpoint holding the `net.*` weights.
    :type checkpoint_path: Path

    :returns: The restorer, and the checkpoint's epoch and global step.
    :rtype: tuple[VoiceRestorer, int | None, int | None]
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
    return net, checkpoint.get("epoch"), checkpoint.get("global_step")


def build_crossfade_weights(
    window: int,
    overlap: int,
    fade_in: bool,
    fade_out: bool,
) -> torch.Tensor:
    """
    Build crossfading weights for overlap-add.

    :param window: Window length in samples.
    :type window: int
    :param overlap: Overlap between consecutive windows in samples.
    :type overlap: int
    :param fade_in: Whether the window's start is overlapped by the previous one.
    :type fade_in: bool
    :param fade_out: Whether the window's end is overlapped by the next one.
    :type fade_out: bool

    :returns: Weights with shape [window].
    :rtype: torch.Tensor
    """

    weights = torch.ones(window)
    if overlap == 0:
        return weights
    ramp = torch.sin(0.5 * math.pi * (torch.arange(overlap) + 0.5) / overlap) ** 2
    if fade_in:
        weights[:overlap] = ramp
    if fade_out:
        weights[-overlap:] = ramp.flip(0)
    return weights


def restore(
    net: VoiceRestorer,
    audio: torch.Tensor,
    window: int,
    overlap: int,
    batch_size: int,
    device: str,
    sampling_kwargs: dict,
) -> torch.Tensor:
    """
    Restore mono audio of any length by overlap-adding fixed-length windows.

    :param net: The restorer.
    :type net: VoiceRestorer
    :param audio: Audio with shape [num_samples], at the model's sample rate.
    :type audio: torch.Tensor
    :param window: Window length in samples.
    :type window: int
    :param overlap: Overlap between consecutive windows in samples.
    :type overlap: int
    :param batch_size: How many windows to restore per forward pass.
    :type batch_size: int
    :param device: Device to run the model on.
    :type device: str
    :param sampling_kwargs: Passed to `restore_audio` (sampling steps, guidance and temperature).
    :type sampling_kwargs: dict

    :returns: Restored audio with shape [num_samples].
    :rtype: torch.Tensor
    """

    num_samples = audio.shape[-1]
    hop = window - overlap
    num_windows = max(1, math.ceil((num_samples - overlap) / hop))
    padded = torch.nn.functional.pad(audio, (0, (num_windows - 1) * hop + window - num_samples))
    windows = padded.unfold(0, window, hop)

    output = torch.zeros_like(padded)
    for start in range(0, num_windows, batch_size):
        batch = windows[start : start + batch_size].unsqueeze(1).to(device)
        with torch.inference_mode():
            restored = net.restore_audio(batch, **sampling_kwargs)
        restored = restored.squeeze(1).float().cpu()
        for offset, chunk in enumerate(restored):
            index = start + offset
            weights = build_crossfade_weights(window, overlap, fade_in=index > 0, fade_out=index < num_windows - 1)
            output[index * hop : index * hop + window] += chunk * weights

    return output[:num_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    args = parser.parse_args()

    print(f"Experiment: {args.experiment}")
    print(f"Checkpoint: {args.checkpoint}")

    net, epoch, global_step = load_net(load_config(args.experiment), args.checkpoint)
    net = net.to(args.device).eval()
    sample_rate = net.sample_rate
    frame_hop = round(sample_rate / net.encoder_frame_rate)
    print(f"  epoch {epoch}, global step {global_step}, {sample_rate} Hz, on {args.device}")

    # Snap both lengths to the frame hop, so every window starts on the latent grid
    window = max(1, round(args.window * sample_rate / frame_hop)) * frame_hop
    overlap = round(args.overlap * sample_rate / frame_hop) * frame_hop
    if 2 * overlap > window:
        raise SystemExit(f"--overlap ({args.overlap} s) must be at most half of --window ({args.window} s).")
    print(f"  windows of {window / sample_rate:.2f} s, overlapping by {overlap / sample_rate:.2f} s")

    steps = args.num_sampling_steps or net.num_sampling_steps
    guidance = args.guidance_scale if args.guidance_scale is not None else net.guidance_scale
    if args.temperature <= 0.0:
        raise SystemExit(f"--temperature must be positive, got {args.temperature}.")
    if args.guidance_target == "ssl":
        if net.ssl_conditioner is None:
            raise SystemExit("--guidance-target ssl needs a model with an SSL conditioner.")
        if guidance == 1.0:
            raise SystemExit("--guidance-target ssl does nothing at a guidance scale of 1.0; pass --guidance-scale.")
    sampling_kwargs = {
        "num_sampling_steps": steps,
        "guidance_scale": guidance,
        "temperature": args.temperature,
        "guidance_target": args.guidance_target,
    }
    target = " (SSL stream only)" if args.guidance_target == "ssl" else ""
    print(f"  {steps} Euler steps, CFG scale {guidance}{target}, Gaussian temperature {args.temperature}")

    paths = sorted(p for p in args.input.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No audio files ({', '.join(sorted(AUDIO_EXTENSIONS))}) under {args.input}.")
    print(f"Restoring {len(paths)} file(s) to {args.output}")

    for path in paths:
        audio, file_sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if file_sample_rate != sample_rate:
            audio = soxr.resample(audio, file_sample_rate, sample_rate)
        peak = np.abs(audio).max()
        if peak < 1e-4:
            print(f"  {path.relative_to(args.input)}: (near-)silent, skipped")
            continue
        audio = torch.from_numpy(audio / peak * INPUT_PEAK).float()

        if args.seed is not None:
            torch.manual_seed(args.seed)
        restored = restore(net, audio, window, overlap, args.batch_size, args.device, sampling_kwargs)

        destination = (args.output / path.relative_to(args.input)).with_suffix(".wav")
        destination.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(destination, restored.clamp(-1.0, 1.0).numpy(), sample_rate, subtype="PCM_24")
        print(f"  {path.relative_to(args.input)}: {audio.shape[-1] / sample_rate:.1f}s")

    print("Done.")


if __name__ == "__main__":
    main()
