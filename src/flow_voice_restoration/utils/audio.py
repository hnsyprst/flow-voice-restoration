from pathlib import Path

import numpy as np
import torch
from soundfile import SoundFile


def audioread(
    path: str | Path,
    audio_duration: float | None = None,
    read_offset_fraction: float | None = None,
    prevent_read_past_eof: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    Read an audio file from the disk into a Tensor.

    :param path: Path to audio file to read.
    :type path: Union[str, Path]
    :param audio_duration: Duration in seconds to read. If `None`, reads the entire file.
        If longer than the file duration, the output is right-padded with near-zero noise.
    :type audio_duration: Optional[float]
    :param read_offset_fraction: Fraction of the file at which to begin reading, in the range `[0, 1)`.
        If `None`, reading starts at the beginning of the file.
        Has no effect when `audio_duration` is longer than the file.
    :type read_offset_fraction: Optional[float]
    :param prevent_read_past_eof: If `True`, prevents reading past the end of the file when `audio_duration`
        and `read_offset_fraction` is set by calculating the read offset based on
        `(file length - audio_duration) * read_offset_fraction`.
        If `False`, will pad reads past EOF with near-zero noise.
        Has no effect when `audio_duration` is `None`.
    :type prevent_read_past_eof: bool
    :return: Tuple of audio tensor with shape (channels, samples) and sample rate.
    :rtype: Tuple[torch.Tensor, int]
    """

    file = SoundFile(path)
    num_output_frames = int(audio_duration) * file.samplerate if audio_duration else file.frames

    # Seek to the read offset if the file is longer than audio duration
    if audio_duration and file.frames > num_output_frames and read_offset_fraction:
        max_frames = file.frames - num_output_frames if prevent_read_past_eof and audio_duration else file.frames
        offset_frames = int(max_frames * read_offset_fraction)
        file.seek(offset_frames)
        # Don't pad the file unnecessarily if `audio_duration` is not set
        if not audio_duration:
            num_output_frames -= offset_frames

    out = np.random.normal(size=(num_output_frames, file.channels)).astype(np.float32) * 1e-8
    file.read(num_output_frames, out=out)
    return torch.from_numpy(out).T, file.samplerate
