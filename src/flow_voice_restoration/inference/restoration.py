import math
from collections.abc import Callable

import torch

from flow_voice_restoration.models.components.voice_restoration import VoiceRestorer


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


def count_windows(
    num_samples: int,
    window: int,
    overlap: int,
) -> int:
    """
    Count the windows `restore` cuts audio into.

    :param num_samples: Length of the audio in samples.
    :type num_samples: int
    :param window: Window length in samples.
    :type window: int
    :param overlap: Overlap between consecutive windows in samples.
    :type overlap: int

    :returns: Number of windows.
    :rtype: int
    """

    return max(1, math.ceil((num_samples - overlap) / (window - overlap)))


def restore(
    net: VoiceRestorer,
    audio: torch.Tensor,
    window: int,
    overlap: int,
    batch_size: int,
    device: str,
    sampling_kwargs: dict,
    model_input_peak: float = 0.9,
    callback: Callable[[int, dict], None] | None = None,
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
    :param model_input_peak: The peak amplitude the training clips were normalised to.
        Input audio will be peak-normalised to this value. Defaults to 0.9.
    :type model_input_peak: float
    :param callback: Called after every Euler step with the index of the batch of windows being
        restored and the sampler's step dict (see `VoiceRestorer.sample`).
    :type callback: Callable[[int, dict], None] | None

    :returns: Restored audio with shape [num_samples].
    :rtype: torch.Tensor
    """

    peak = torch.abs(audio).max()
    audio = audio / peak * model_input_peak

    num_samples = audio.shape[-1]
    hop = window - overlap
    num_windows = count_windows(num_samples, window, overlap)
    padded = torch.nn.functional.pad(audio, (0, (num_windows - 1) * hop + window - num_samples))
    windows = padded.unfold(0, window, hop)

    output = torch.zeros_like(padded)
    for batch_index, start in enumerate(range(0, num_windows, batch_size)):
        batch = windows[start : start + batch_size].unsqueeze(1).to(device)
        step_callback = None if callback is None else lambda d, b=batch_index: callback(b, d)
        with torch.inference_mode():
            restored = net.restore_audio(batch, **sampling_kwargs, callback=step_callback)
        restored = restored.squeeze(1).float().cpu()
        for offset, chunk in enumerate(restored):
            index = start + offset
            weights = build_crossfade_weights(window, overlap, fade_in=index > 0, fade_out=index < num_windows - 1)
            output[index * hop : index * hop + window] += chunk * weights

    return output[:num_samples]
