# Ported from Sidon: https://github.com/sarulab-speech/Sidon/blob/main/src/sidon/data/preprocess/functional_degrations.py

import random
from collections.abc import Iterator, Sequence
from itertools import chain
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import torch
import torchaudio
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from flow_voice_restoration.data.components.rir_utils import estimate_early_rir
from flow_voice_restoration.utils.audio import audioread

# (format, bit_rate) presets for `codec`. Formats are torchcodec/FFmpeg muxer names:
# "adts" is AAC in an ADTS stream, "ogg" is Vorbis.
DEFAULT_CODEC_PRESETS: list[tuple[str, int]] = [
    ("mp3", 16_000),
    ("mp3", 32_000),
    ("mp3", 64_000),
    ("mp3", 96_000),
    ("mp3", 128_000),
    ("opus", 6_000),
    ("opus", 12_000),
    ("opus", 24_000),
    ("opus", 48_000),
    ("adts", 16_000),
    ("adts", 32_000),
    ("adts", 64_000),
    ("adts", 96_000),
    ("ogg", 32_000),
    ("ogg", 64_000),
]


class RandomAudioSampler:
    """
    Infinite iterator yielding `(audio, sample_rate)` random crops from a directory tree of audio
    files. Stands in for Sidon's WebDataset noise/RIR streams as the `noise_ds`/`rir_ds` arguments.
    Uses the global `random` state, so DataLoader workers (which are seeded per worker) diverge.
    """

    def __init__(
        self,
        data_dirs: str | Path | Sequence[str | Path],
        duration: float | None = 10.0,
        supported_file_extensions: Sequence[str] = ("wav", "flac"),
    ):
        """
        Initialise the sampler.

        :param data_dirs: Directory (or directories) to recursively glob audio files from.
        :type data_dirs: Union[str, Path, Sequence[Union[str, Path]]]
        :param duration: Crop duration in seconds; files longer than this are cropped at a random
            offset, shorter files are right-padded with near-zero noise (see `audioread`).
            If `None`, whole files are returned (use this for impulse responses).
        :type duration: Optional[float]
        :param supported_file_extensions: Audio file extensions to read, without the dot.
        :type supported_file_extensions: Sequence[str]
        """

        if isinstance(data_dirs, str | Path):
            data_dirs = [data_dirs]
        self.paths = sorted(
            chain(
                *[
                    Path(data_dir).rglob(f"*.{extension}")
                    for data_dir in data_dirs
                    for extension in supported_file_extensions
                ]
            )
        )
        if not self.paths:
            raise ValueError(
                f"Did not find any audio files under {data_dirs} with extensions {supported_file_extensions}"
            )
        self.duration = duration

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int]]:
        return self

    def __next__(self) -> tuple[torch.Tensor, int]:
        path = random.choice(self.paths)
        return audioread(path, audio_duration=self.duration, read_offset_fraction=random.random())


# align two waveforms by shifting the codec output based on mse, then pad/crop to the original length
def align_codec_waveform(original: torch.Tensor, codec_applied: torch.Tensor) -> torch.Tensor:
    if original.size(1) != codec_applied.size(1):
        if original.size(1) < codec_applied.size(1):
            # Decoded audio is longer (e.g. AAC priming samples at the start):
            # find the start index in the decoded audio which minimizes the mse against the original
            mse = [
                torch.mean((codec_applied[:, i : i + original.size(1)] - original) ** 2)
                for i in range(codec_applied.size(1) - original.size(1) + 1)
            ]
            start_index = int(torch.argmin(torch.tensor(mse)))
            codec_applied = codec_applied[:, start_index : start_index + original.size(1)]
        else:
            # Decoded audio is shorter (e.g. truncated final frame):
            # find the start index in the original which minimizes the mse, then shift to match
            mse = [
                torch.mean((original[:, i : i + codec_applied.size(1)] - codec_applied) ** 2)
                for i in range(original.size(1) - codec_applied.size(1) + 1)
            ]
            start_index = int(torch.argmin(torch.tensor(mse)))
            codec_applied = torch.nn.functional.pad(
                codec_applied, (start_index, original.size(1) - codec_applied.size(1) - start_index)
            )
    return codec_applied


@torch.inference_mode()
def codec(
    x: torch.Tensor,
    sr: int,
    codec_presets: Sequence[tuple[str, int]] = tuple(DEFAULT_CODEC_PRESETS),
) -> torch.Tensor:
    x = x.view(1, -1)
    audio_format, bit_rate = random.choice(list(codec_presets))
    try:
        encoded = AudioEncoder(x, sample_rate=sr).to_tensor(format=audio_format, bit_rate=bit_rate)
        codec_applied = AudioDecoder(encoded).get_all_samples()
    except Exception:
        return x
    decoded = codec_applied.data.mean(0, keepdim=True)
    if codec_applied.sample_rate != sr:
        decoded = torchaudio.functional.resample(decoded, codec_applied.sample_rate, sr)
    return align_codec_waveform(x, codec_applied=decoded)


def packet_loss(x: torch.Tensor, sr: int, loss_rate: float = 0.1) -> torch.Tensor:
    # Randomly mask ~`loss_rate` of the audio with chunks of length 20 to 200 ms
    x = x.view(1, -1).clone()
    total_duration = x.size(1) / sr
    num_chunks = int(total_duration * 3 * loss_rate)

    for _ in range(num_chunks):
        chunk_duration_ms = random.uniform(20, 200)
        start_time = random.uniform(0, total_duration - chunk_duration_ms / 1000)
        start_sample = int(start_time * sr)
        end_sample = int((start_time + chunk_duration_ms / 1000) * sr)
        x[:, start_sample:end_sample] = 0
    return x


@torch.inference_mode()
def band_limit(
    x: torch.Tensor,
    sr: int,
    candidate_srs: Sequence[int] = (8000, 16000, 22050, 24000, 44100, 48000),
) -> torch.Tensor:
    x = x.view(1, -1)
    target_sr = random.sample(list(candidate_srs), 1)[0]
    down_sampled = torchaudio.functional.resample(x, sr, target_sr)
    up_sampled = torchaudio.functional.resample(down_sampled, target_sr, sr)
    # Resampling can change the length by a sample; pad/crop back to keep the pair aligned
    return torch.nn.functional.pad(up_sampled, (0, max(0, x.size(1) - up_sampled.size(1))))[:, : x.size(1)]


@torch.inference_mode()
def add_non_parametric_noise(
    x: torch.Tensor,
    sr: int,
    noise_ds: Iterator[tuple[torch.Tensor, int]],
    snr_range: tuple[float, float] = (-5.0, 20.0),
) -> torch.Tensor:
    noise, noise_sr = next(noise_ds)
    if noise.ndim == 1:
        noise = noise.unsqueeze(0)
    noise = noise[0].unsqueeze(0)
    x = x.view(1, -1)
    noise = torchaudio.functional.resample(noise, noise_sr, sr)
    if noise.abs().max() < 1e-6:
        # Near-silent noise crop (e.g. a "quiet room" recording): adding it at a target SNR would
        # blow it up to match the SNR or produce NaNs, so skip
        return x
    noise = noise.repeat(1, x.size(1) // noise.size(1) + 1)[:, : x.size(1)]
    return torchaudio.functional.add_noise(x, noise, snr=torch.tensor(random.uniform(*snr_range)).unsqueeze(0))


@torch.inference_mode()
def add_noise_at_level(
    x: torch.Tensor,
    sr: int,
    noise_ds: Iterator[tuple[torch.Tensor, int]],
    level_dbfs_range: tuple[float, float] = (-40.0, -10.0),
) -> torch.Tensor:
    """
    Add a noise crop scaled to an absolute RMS level, rather than to a level relative to `x`.

    `add_non_parametric_noise` mixes at a signal-to-noise ratio, which means nothing for a silent
    signal: with no signal energy, the noise is scaled to nothing. This sets the noise's own level, so
    it can stand alone as the input of a clip whose target is silence.

    :param x: Audio tensor of shape `(channels, samples)` (mono); may be all zeros.
    :type x: torch.Tensor
    :param sr: Sample rate of `x` in Hz.
    :type sr: int
    :param noise_ds: Iterator yielding `(noise_audio, sample_rate)` tuples (see `RandomAudioSampler`).
    :type noise_ds: Iterator[Tuple[torch.Tensor, int]]
    :param level_dbfs_range: Range the noise's RMS level is drawn from, in dB relative to full scale.
    :type level_dbfs_range: Tuple[float, float]
    :return: `x` plus the scaled noise, same shape as `x`. `x` unchanged if the noise crop is silent.
    :rtype: torch.Tensor
    """

    noise, noise_sr = next(noise_ds)
    if noise.ndim == 1:
        noise = noise.unsqueeze(0)
    noise = noise[0].unsqueeze(0)
    x = x.view(1, -1)
    noise = torchaudio.functional.resample(noise, noise_sr, sr)
    if noise.abs().max() < 1e-6:
        # As in `add_non_parametric_noise`: a near-silent crop would be blown up to the target level.
        return x
    noise = noise.repeat(1, x.size(1) // noise.size(1) + 1)[:, : x.size(1)]
    target_rms = 10.0 ** (random.uniform(*level_dbfs_range) / 20.0)
    return x + noise * (target_rms / noise.pow(2).mean().sqrt())


def clipping(speech_sample, min_quantile: float = 0.0, max_quantile: float = 0.9):
    """Apply the clipping distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        min_quantile (float): lower bound on the quantile of samples to be clipped
        max_quantile (float): upper bound on the quantile of samples to be clipped

    Returns:
        ret (np.ndarray): clipped speech sample (1, Time)
    """
    q = np.array([min_quantile, max_quantile])
    min_, max_ = np.quantile(speech_sample, q, axis=-1, keepdims=False)
    # per-channel clipping
    ret = np.stack(
        [np.clip(speech_sample[i], min_[i], max_[i]) for i in range(speech_sample.shape[0])],
        axis=0,
    )
    return ret


def clip(x: torch.Tensor) -> torch.Tensor:
    x = x.view(1, -1)
    min_q = random.uniform(0.0, 0.1)
    max_q = random.uniform(0.9, 1.0)
    # numpy's quantiles are float64, which would silently promote the audio; keep the input dtype
    return torch.tensor(clipping(x.numpy(), min_quantile=min_q, max_quantile=max_q), dtype=x.dtype)


@torch.inference_mode()
def convolve_rir_pra(
    x: torch.Tensor,
    sr: int,
    rt60_range: tuple[float, float] = (0.1, 2.0),
    room_dim_range: tuple[float, float] = (2.0, 20.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate room reverberation with a pyroomacoustics ShoeBox room.

    Returns a `(direct, reverberant)` pair: the direct-path (anechoic) signal should replace the
    clean signal so the pair stays time-aligned.
    """
    x = x.view(1, -1)
    rt60 = random.uniform(*rt60_range)
    while True:
        try:
            room_dim = (
                random.uniform(*room_dim_range),
                random.uniform(*room_dim_range),
                random.uniform(*room_dim_range),
            )
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            break
        except ValueError:
            continue
    room = pra.ShoeBox(
        room_dim,
        fs=sr,
        max_order=max_order,
        materials=pra.Material(e_absorption),
    )
    margin = 0.01
    source_pos = [
        random.uniform(margin, room_dim[0] - margin),
        random.uniform(margin, room_dim[1] - margin),
        random.uniform(margin, room_dim[2] - margin),
    ]
    room.add_source(source_pos)
    mic_pos = [
        random.uniform(margin, room_dim[0] - margin),
        random.uniform(margin, room_dim[1] - margin),
        random.uniform(margin, room_dim[2] - margin),
    ]
    room.add_microphone(mic_pos)
    anechoic_room = pra.ShoeBox(
        room_dim,
        fs=sr,
        max_order=0,
        materials=pra.Material(1.0, scattering=0.0),
    )
    anechoic_room.add_source(source_pos)
    anechoic_room.add_microphone(mic_pos)
    room.compute_rir()
    anechoic_room.compute_rir()
    # `compute_rir` fills in `rir`, which pyroomacoustics initialises to `None`
    assert room.rir is not None and anechoic_room.rir is not None
    rir = torch.tensor(room.rir[0][0]).unsqueeze(0).float()
    direct_rir = torch.tensor(anechoic_room.rir[0][0]).unsqueeze(0).float()
    original_maximum_amp = x.abs().max()
    rir_convolved = torchaudio.functional.fftconvolve(x, rir)[:, : x.size(1)]
    direct_rir_convolved = torchaudio.functional.fftconvolve(x, direct_rir)[:, : x.size(1)]
    convolved_maximum_amp = max(rir_convolved.abs().max().item(), direct_rir_convolved.abs().max().item())
    rir_convolved = (rir_convolved / convolved_maximum_amp) * original_maximum_amp
    direct_rir_convolved = (direct_rir_convolved / convolved_maximum_amp) * original_maximum_amp
    return direct_rir_convolved, rir_convolved


@torch.inference_mode()
def convolve_rir(
    x: torch.Tensor,
    sr: int,
    rir_ds: Iterator[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convolve with a measured RIR.

    Returns a `(direct, reverberant)` pair: the early-reflection (first 50 ms) signal should
    replace the clean signal so the pair stays time-aligned.
    """
    x = x.view(1, -1)
    rir, rir_sr = next(rir_ds)
    rir = rir[random.randint(0, rir.shape[0] - 1)].unsqueeze(0)
    rir = rir.view(1, -1)
    rir = torchaudio.functional.resample(rir, rir_sr, sr)
    direct_rir = torch.tensor(estimate_early_rir(rir.numpy(), early_rir_sec=0.05, fs=sr))
    rir_convolved = torchaudio.functional.fftconvolve(x, rir)[:, : x.size(1)]
    direct_rir_convolved = torchaudio.functional.fftconvolve(x, direct_rir)[:, : x.size(1)]
    return direct_rir_convolved, rir_convolved


@torch.inference_mode()
def lowcut(x: torch.Tensor, sr: int, cutoff: float = 50) -> torch.Tensor:
    return torchaudio.functional.highpass_biquad(x, sr, cutoff)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x / (x.abs().max() + 1e-7)) * 0.9


@torch.inference_mode()
def degrade(
    clean: torch.Tensor,
    sr: int,
    noise_ds: Iterator[tuple[torch.Tensor, int]],
    prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the degradation chain (Sidon's `WebDatasetDataModule.add_noise_pipeline`), applying each
    degradation with probability `prob`.

    :param clean: Clean audio tensor of shape `(channels, samples)` (mono).
    :type clean: torch.Tensor
    :param sr: Sample rate of `clean` in Hz.
    :type sr: int
    :param noise_ds: Iterator yielding `(noise_audio, sample_rate)` tuples (see `RandomAudioSampler`).
    :type noise_ds: Iterator[Tuple[torch.Tensor, int]]
    :param prob: Probability of applying each degradation.
    :type prob: float
    :return: `(clean, noisy)` pair of equal length. If reverb was applied, the returned clean
        signal is the direct-path signal, keeping the pair time-aligned.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    clean = clean.view(1, -1)
    noisy = clean
    if random.random() < prob:
        clean, noisy = convolve_rir_pra(noisy, sr)
    if random.random() < prob:
        noisy = add_non_parametric_noise(noisy, sr, noise_ds)
    if random.random() < prob:
        noisy = band_limit(noisy, sr)
    if random.random() < prob:
        noisy = clip(noisy)
    if random.random() < prob:
        noisy = codec(noisy, sr)
    if random.random() < prob:
        noisy = packet_loss(noisy, sr)
    return clean, torch.nan_to_num(noisy)


@torch.inference_mode()
def degrade_silence(
    num_samples: int,
    sr: int,
    noise_ds: Iterator[tuple[torch.Tensor, int]],
    prob: float = 0.5,
    level_dbfs_range: tuple[float, float] = (-40.0, -10.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Make a clip with nothing to restore: noise alone as the input, digital silence as the target.

    Every target `degrade` produces holds a voice, so a model trained only on those learns that there
    is always one to recover -- and finds one in drums or traffic. These pairs teach it that the
    answer can be silence.

    The chain follows `degrade` with the voice left out. The noise is always added, at an absolute
    level (`add_noise_at_level`), since there is no signal to set a ratio against; reverb then
    colours the noise, as a room colours everything recorded in it; the rest apply as in `degrade`,
    each with probability `prob`. Reverb runs on the noise rather than the silent target:
    `convolve_rir_pra` rescales by the input's peak, which is 0/0 for silence.

    :param num_samples: Length of the clip in samples.
    :type num_samples: int
    :param sr: Sample rate in Hz.
    :type sr: int
    :param noise_ds: Iterator yielding `(noise_audio, sample_rate)` tuples (see `RandomAudioSampler`).
    :type noise_ds: Iterator[Tuple[torch.Tensor, int]]
    :param prob: Probability of applying each degradation after the noise.
    :type prob: float
    :param level_dbfs_range: Range the noise's RMS level is drawn from, in dBFS.
    :type level_dbfs_range: Tuple[float, float]
    :return: `(clean, noisy)`, each of shape `(1, num_samples)`; `clean` is all zeros.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    clean = torch.zeros(1, num_samples)
    noisy = add_noise_at_level(clean, sr, noise_ds, level_dbfs_range)
    if random.random() < prob and noisy.abs().max() > 0:
        _, noisy = convolve_rir_pra(noisy, sr)
    if random.random() < prob:
        noisy = band_limit(noisy, sr)
    if random.random() < prob:
        noisy = clip(noisy)
    if random.random() < prob:
        noisy = codec(noisy, sr)
    if random.random() < prob:
        noisy = packet_loss(noisy, sr)
    return clean, torch.nan_to_num(noisy)
