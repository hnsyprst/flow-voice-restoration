from pathlib import Path

import torch
import torchaudio
from torchaudio.functional import resample
from transformers import Wav2Vec2BertModel


def kaldi_fbank(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> torch.Tensor:
    """
    Compute the log-mel filterbank of one waveform, as w2v-BERT 2.0's feature extractor does.

    :param waveform: Tensor containing a waveform with shape [channels, num_samples].
    :type waveform: torch.Tensor
    :param sample_rate: Sample rate of `waveform`.
    :type sample_rate: int
    :param frame_length_ms: Length of each filterbank frame, in milliseconds.
    :type frame_length_ms: float
    :param frame_shift_ms: Hop between filterbank frames, in milliseconds.
    :type frame_shift_ms: float

    :returns: Log-mel filterbank with shape [num_frames, 80].
    :rtype: torch.Tensor
    """

    return torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=frame_length_ms,
        frame_shift=frame_shift_ms,
        sample_frequency=float(sample_rate),
        low_freq=20.0,
        high_freq=0.0,
        window_type="povey",
        preemphasis_coefficient=0.97,
        remove_dc_offset=True,
        round_to_power_of_two=True,
        snip_edges=True,
        use_energy=False,
        dither=0.0,
    )


def input_features(
    audio: torch.Tensor,
    input_sample_rate: int,
    w2v_bert_sample_rate: int = 16000,
    feature_stride: int = 2,
) -> torch.Tensor:
    """
    Compute w2v-BERT 2.0's input features from a batch of audio, differentiably.

    :param audio: Audio with shape [B, num_samples] or [B, 1, num_samples] at `input_sample_rate`.
    :type audio: torch.Tensor
    :param input_sample_rate: Sample rate of `audio`.
    :type input_sample_rate: int
    :param w2v_bert_sample_rate: Sample rate to resample to before feature extraction.
    :type w2v_bert_sample_rate: int
    :param feature_stride: Number of consecutive filterbank frames to stack into one model input frame.
    :type feature_stride: int

    :returns: Input features with shape [B, num_frames // feature_stride, 160], at 50 frames per second.
    :rtype: torch.Tensor
    """

    if audio.ndim == 3:
        audio = audio[:, 0]

    audio = resample(audio, input_sample_rate, w2v_bert_sample_rate)
    # 16-bit signed integer scale to comply with Kaldi's fbank
    audio = audio * (2**15)
    features = torch.stack([kaldi_fbank(item[None]) for item in audio])
    mean = features.mean(dim=1, keepdim=True)
    variance = features.var(dim=1, keepdim=True)
    features = (features - mean) / torch.sqrt(variance + 1e-7)

    batch_size, num_frames, num_mel_bins = features.shape
    num_frames -= num_frames % feature_stride
    return features[:, :num_frames].reshape(batch_size, num_frames // feature_stride, num_mel_bins * feature_stride)


def load_denoised_w2v_bert(
    checkpoint_path: str | Path,
    model_name: str = "facebook/w2v-bert-2.0",
    num_hidden_layers: int = 8,
    lora_rank: int = 64,
    lora_alpha: int = 16,
) -> Wav2Vec2BertModel:
    """
    Load Sidon's denoising w2v-BERT student from a `FeaturePredictorLightningModule` checkpoint.

    :param checkpoint_path: Path to the Lightning checkpoint.
    :type checkpoint_path: Union[str, Path]
    :param model_name: Hub id (or local path) of the base checkpoint the student was adapted from.
    :type model_name: str
    :param num_hidden_layers: Encoder layers the student was truncated to.
    :type num_hidden_layers: int
    :param lora_rank: Rank the adapters were trained at.
    :type lora_rank: int
    :param lora_alpha: Scaling numerator the adapters were trained with.
    :type lora_alpha: int

    :returns: The student, frozen and in eval mode.
    :rtype: Wav2Vec2BertModel
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prefix = "student_ssl_model."
    student = {key[len(prefix) :]: value for key, value in checkpoint["state_dict"].items() if key.startswith(prefix)}
    if not student:
        raise ValueError(f"{checkpoint_path} holds no `{prefix}*` weights; is it a Sidon feature predictor checkpoint?")

    scaling = lora_alpha / lora_rank
    merged = {key.replace("base_layer.", ""): value.clone() for key, value in student.items() if "lora_" not in key}
    for key, down_projection in student.items():
        if "lora_A" not in key:
            continue
        up_projection = student[key.replace("lora_A", "lora_B")]
        target = key.replace("lora_A.default.weight", "weight").replace("base_layer.", "")
        merged[target] = merged[target] + scaling * (up_projection @ down_projection)

    model = Wav2Vec2BertModel.from_pretrained(model_name, num_hidden_layers=num_hidden_layers, layerdrop=0.0)
    model.load_state_dict(merged, strict=True)
    return model.requires_grad_(False).eval()
