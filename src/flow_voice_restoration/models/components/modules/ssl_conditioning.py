from pathlib import Path

import torch
import torch.nn as nn
from transformers import Wav2Vec2BertModel

from flow_voice_restoration.utils.w2v_bert import (
    input_features,
    load_denoised_w2v_bert,
)


class DenoisedSslConditioner(nn.Module):
    """
    Per-frame conditioning from Sidon's denoising w2v-BERT model.

    Its features are matched to the latent frame rate and projected to the conditioning width,
    to be added to the encoder's own conditioning.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        conditioning_dimension: int,
        input_sample_rate: int,
        num_sub_windows: int = 1,
        model_name: str = "facebook/w2v-bert-2.0",
        num_hidden_layers: int = 8,
        feature_stride: int = 2,
        fbank_frame_length_ms: int = 25,
        fbank_frame_shift_ms: int = 10,
    ):
        """
        Initialise the conditioner.

        :param checkpoint_path: Path to the Sidon feature predictor checkpoint holding the denoised w2v-BERT model.
        :type checkpoint_path: str | Path
        :param conditioning_dimension: Width of the conditioning the velocity model consumes.
        :type conditioning_dimension: int
        :param input_sample_rate: Sample rate of the audio passed to `forward`.
            It is resampled from this rate to the 16 kHz w2v-BERT expects.
        :type input_sample_rate: int
        :param num_sub_windows: How many parts each latent frame's window is split into before averaging.
            1 averages the whole window into one vector; 2 keeps an early and a late half,
            which the projection sees side by side.
        :type num_sub_windows: int
        :param model_name: Hub id (or local path) of the base checkpoint the denoised w2v-BERT model was adapted from.
        :type model_name: str
        :param num_hidden_layers: Encoder layers the denoised w2v-BERT model was truncated to.
        :type num_hidden_layers: int
        """

        super().__init__()

        if num_sub_windows < 1:
            raise ValueError(f"`num_sub_windows` must be at least 1, got {num_sub_windows}.")

        model = load_denoised_w2v_bert(checkpoint_path, model_name=model_name, num_hidden_layers=num_hidden_layers)
        # Little hack, we make the model a one-element list so its not registered as a parameter of this module,
        # so its parameters are not registered as parameters of this module
        self._model = [model]

        self.input_sample_rate = input_sample_rate
        self.num_sub_windows = num_sub_windows
        self.feature_stride = feature_stride
        # Seconds between consecutive denoised w2v-BERT frames
        self.frame_hop = self.feature_stride * fbank_frame_shift_ms / 1000.0
        # Centre of denoised w2v-BERT frame 0, in seconds
        self.frame_center = ((self.feature_stride - 1) * fbank_frame_shift_ms + fbank_frame_length_ms) / 2 / 1000.0

        self.norm = nn.LayerNorm(model.config.hidden_size)
        # We zero-initialise the projection so models can be warm-started without the conditioning,
        # and fine-tuned to use the conditioning in post-training
        self.projection = nn.Linear(model.config.hidden_size * self.num_sub_windows, conditioning_dimension)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    @property
    def model(self) -> Wav2Vec2BertModel:
        """
        The frozen denoising w2v-BERT model, kept out of this module's parameters and buffers.
        """

        return self._model[0]

    def stack_aligned_sub_windows(
        self,
        features: torch.Tensor,
        num_frames: int,
        frame_rate: float,
    ) -> torch.Tensor:
        """
        Resample a fine-grained feature sequence onto a coarser frame grid.

        :param features: Fine-grained features with shape [B, T, D].
        :type features: torch.Tensor
        :param num_frames: Number of output frames.
        :type num_frames: int
        :param frame_rate: Output frames per second.
        :type frame_rate: float

        :returns: Features with shape [B, num_frames, D * num_sub_windows].
        :rtype: torch.Tensor
        """

        batch_size, num_feature_frames, feature_dimension = features.shape
        sub_windows = torch.arange(num_frames * self.num_sub_windows, device=features.device, dtype=torch.float64)
        times = (sub_windows + 0.5) / (self.num_sub_windows * frame_rate)
        positions = ((times - self.frame_center) / self.frame_hop).clamp(0, num_feature_frames - 1)

        lower = positions.floor().long()
        upper = (lower + 1).clamp(max=num_feature_frames - 1)
        weight = (positions - lower).to(features.dtype)[None, :, None]
        interpolated = features[:, lower] * (1.0 - weight) + features[:, upper] * weight
        return interpolated.reshape(batch_size, num_frames, self.num_sub_windows * feature_dimension)

    def forward(
        self,
        audio: torch.Tensor,
        num_frames: int,
        frame_rate: float,
    ) -> torch.Tensor:
        """
        Describe degraded audio at the latent frame rate and the conditioning width, to be added to
        the encoder's own conditioning.

        :param audio: Degraded audio with shape [B, 1, num_samples] at `input_sample_rate`.
        :type audio: torch.Tensor
        :param num_frames: Latent frames to produce conditioning for.
        :type num_frames: int
        :param frame_rate: Latent frames per second.
        :type frame_rate: float

        :returns: Conditioning features with shape [B, num_frames, conditioning_dimension].
        :rtype: torch.Tensor
        """

        parameter = next(self.model.parameters())
        with torch.no_grad():
            features = input_features(audio, self.input_sample_rate, feature_stride=self.feature_stride)
            if parameter.device != features.device:
                self.model.to(device=features.device)  # ty:ignore[missing-argument]
                parameter = next(self.model.parameters())
            hidden_state = self.model(input_features=features.to(parameter.dtype)).last_hidden_state

        features = self.norm(hidden_state.to(self.norm.weight.dtype))

        stacked = self.stack_aligned_sub_windows(features, num_frames, frame_rate)
        return self.projection(stacked)
