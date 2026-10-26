import math
from collections.abc import Callable
from typing import Literal, Self, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from flow_voice_restoration.models.components.modules.dit import DiT
from flow_voice_restoration.models.components.modules.ssl_conditioning import DenoisedSslConditioner
from flow_voice_restoration.utils.autoencoder import ExportedAutoencoder


class VoiceRestorer(nn.Module):
    """
    Flow-matching voice restoration model using a Diffusion Transformer.
    """

    def __init__(
        self,
        autoencoder: ExportedAutoencoder,
        velocity_model: DiT,
        ssl_conditioner: DenoisedSslConditioner | None = None,
        freeze_encoder: bool = False,
        num_sampling_steps: int = 32,
        guidance_scale: float = 1.0,
        timestep_shift_base_dimension: int | None = None,
    ):
        """
        Initialize FlowVoiceRestorer.

        :param autoencoder: Autoencoder used to decode predicted latents.
            Its encoder is also used to warm-start the (optionally trainable) conditioning encoder.
            The bottleneck and decoder are frozen during training.
        :type autoencoder: ExportedAutoencoder
        :param velocity_model: DiT predicting flow-matching velocities.
        :type velocity_model: DiT
        :param ssl_conditioner: Optional content conditioning from denoised w2v-BERT features.
            `None` leaves the model's parameters identical to a run without it,
            so its checkpoints load either way.
        :type ssl_conditioner: DenoisedSslConditioner | None
        :param freeze_encoder: Whether to freeze the conditioning encoder and hold it in eval mode during training.
        :type freeze_encoder: bool
        :param num_sampling_steps: Default number of Euler steps used by `sample` and `restore_audio`.
        :type num_sampling_steps: int
        :param guidance_scale: Default classifier-free guidance scale.
            1.0 disables guidance; larger values increase guidance strength.
        :type guidance_scale: float
        :param timestep_shift_base_dimension: Base dimension of the dimension-dependent timestep shift,
            set to 4096 by default in the RAE [1] recipe. `None` disables the shift.
        :type timestep_shift_base_dimension: int | None

        .. [1] https://arxiv.org/abs/2510.11690
        """

        super().__init__()

        if timestep_shift_base_dimension is not None and timestep_shift_base_dimension <= 0:
            raise ValueError(f"{timestep_shift_base_dimension=} must be positive.")

        same = autoencoder.autoencoder
        self.encoder = same.encoder
        self.bottleneck = same.bottleneck
        self.decoder = same.decoder

        self._patchify = same.patchify
        self._depatchify = same.depatchify
        self.patch_size = same.patch_size

        downsampling_ratio = same.patch_size * math.prod(
            cast(int, block.stride) for block in same.encoder.transformer_layers
        )
        self.sample_rate = autoencoder.sample_rate
        self.encoder_frame_rate = autoencoder.sample_rate / downsampling_ratio

        self.freeze_encoder = freeze_encoder
        self.velocity_model = velocity_model
        self.ssl_conditioner = ssl_conditioner

        self.num_sampling_steps = num_sampling_steps
        self.guidance_scale = guidance_scale
        self.timestep_shift_base_dimension = timestep_shift_base_dimension

        self.bottleneck.requires_grad_(False)
        self.decoder.requires_grad_(False)
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def train(
        self,
        mode: bool = True,
    ) -> Self:
        super().train(mode)
        self.bottleneck.eval()
        self.decoder.eval()
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode audio to standardized bottleneck latents.

        :param audio: Audio with shape [B, 1, num_samples].
        :type audio: torch.Tensor

        :returns: Standardized bottleneck latents with shape [B, T, latent_dimension].
        :rtype: torch.Tensor
        """

        audio = F.pad(audio, (0, -audio.shape[-1] % self.patch_size))
        return self.encoder(self._patchify(audio)).swapaxes(1, 2)

    def decode(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode standardized bottleneck latents to audio.

        :param latents: Standardized bottleneck latents with shape [B, T, latent_dimension].
        :type latents: torch.Tensor

        :returns: Audio with shape [B, 1, num_samples].
        :rtype: torch.Tensor
        """

        decoder_inputs = self.bottleneck.decode(latents).swapaxes(1, 2)
        return self._depatchify(self.decoder(decoder_inputs))

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_conditioning: torch.Tensor,
        ssl_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict the flow-matching velocities for a batch of noisy latents.

        :param noisy_latents: Noisy standardized bottleneck latents with shape [B, T, latent_dimension].
        :type noisy_latents: torch.Tensor
        :param timestep: Flow times in [0, 1] with shape [B].
        :type timestep: torch.Tensor
        :param encoder_conditioning: The encoder's conditioning features with shape [B, T, conditioning_dimension].
        :type encoder_conditioning: torch.Tensor
        :param ssl_conditioning: The SSL conditioning features with shape [B, S, ssl_feature_width], or `None`.
        :type ssl_conditioning: torch.Tensor | None
        """

        conditioning = encoder_conditioning + ssl_conditioning if ssl_conditioning is not None else encoder_conditioning
        return self.velocity_model(noisy_latents, timestep, conditioning)

    def get_conditioning(
        self,
        audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Extract the two conditioning streams from degraded audio.

        :param audio: Degraded audio with shape [B, 1, num_samples].
        :type audio: torch.Tensor

        :returns: The encoder's conditioning features with shape [B, T, conditioning_dimension],
            and the SSL stream with shape [B, S, ssl feature width],
            or `None` if this model was constructed without an `ssl_conditioner`.
        :rtype: tuple[torch.Tensor, torch.Tensor | None]
        """

        encoder_conditioning = self._encode(audio)
        if self.ssl_conditioner is None:
            return encoder_conditioning, None
        return encoder_conditioning, self.ssl_conditioner(audio, encoder_conditioning.shape[1], self.encoder_frame_rate)

    def shift_timesteps(
        self,
        timestep: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Apply the dimension-dependent timestep shift from RAE [1], or return `timestep` unchanged if it is off.

        :param timestep: Flow times in [0, 1], any shape.
        :type timestep: torch.Tensor
        :param num_frames: Latent frames per sample.
            Times the latent dimension, this is the effective data dimension m in alpha = sqrt(m / n).
        :type num_frames: int

        :returns: Shifted flow times, same shape as `timestep`.
        :rtype: torch.Tensor

        .. [1] https://arxiv.org/abs/2510.11690
        """

        if self.timestep_shift_base_dimension is None:
            return timestep

        effective_dimension = num_frames * self.velocity_model.latent_dimension
        alpha = (effective_dimension / self.timestep_shift_base_dimension) ** 0.5
        return timestep / (1.0 + (alpha - 1.0) * (1.0 - timestep))

    @torch.inference_mode()
    def sample(
        self,
        encoder_conditioning: torch.Tensor,
        num_sampling_steps: int | None = None,
        guidance_scale: float | None = None,
        ssl_conditioning: torch.Tensor | None = None,
        temperature: float = 1.0,
        guidance_target: Literal["all", "ssl"] = "all",
        callback: Callable[[dict], None] | None = None,
    ) -> torch.Tensor:
        """
        Sample clean standardized latents by Euler-integrating the learned velocity field,
        from Gaussian noise (t=0) to data (t=1).

        Features a temperature parameter proposed in CALM [1], which can be used to trade
        diversity for fidelity, analogously to the temperature parameter in autoregressive models.

        :param encoder_conditioning: The encoder's conditioning features with shape [B, T, conditioning_dimension].
        :type encoder_conditioning: torch.Tensor
        :param num_sampling_steps: Number of Euler steps. Defaults to `self.num_sampling_steps`.
        :type num_sampling_steps: int | None
        :param guidance_scale: Guidance scale, defaults to `self.guidance_scale`.
        :type guidance_scale: float | None
        :param ssl_conditioning: The SSL stream with shape [B, T, conditioning_dimension],
            summed into `encoder_conditioning` by `forward`, or `None` for a model without an `ssl_conditioner`.
        :type ssl_conditioning: torch.Tensor | None
        :param temperature: Temperature control from CALM [1].
            The initial noise is drawn with variance `temperature`, i.e. standard deviation `sqrt(temperature)`,
            trading diversity for fidelity below 1.0. 1.0 is plain sampling.
        :type temperature: float
        :param guidance_target: What to take away for the weak branch of classifier-free guidance (CFG).
            `all` zeroes every stream, obtaining the unconditional velocity.
            `ssl` keeps the encoder's features and takes away only the SSL stream.
        :type guidance_target: Literal["all", "ssl"]
        :param callback: Called after every Euler step with a dict holding information about the step.
        :type callback: Callable[[dict], None] | None

        :returns: Sampled latents with shape [B, T, latent_dimension].
        :rtype: torch.Tensor

        .. [1] https://arxiv.org/abs/2509.06926
        """

        num_sampling_steps = num_sampling_steps or self.num_sampling_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.guidance_scale

        if guidance_target not in ["all", "ssl"]:
            raise ValueError(f"`guidance_target` must be one of ['all', 'ssl'], got {guidance_target!r}.")
        if guidance_target == "ssl" and guidance_scale != 1.0 and ssl_conditioning is None:
            raise ValueError("SSL-only guidance needs `ssl_conditioning`; none was given.")

        dtype = (
            encoder_conditioning.dtype
            if ssl_conditioning is None
            else torch.promote_types(encoder_conditioning.dtype, ssl_conditioning.dtype)
        )
        batch_size, sequence_length, _ = encoder_conditioning.shape
        x = torch.randn(
            batch_size,
            sequence_length,
            self.velocity_model.latent_dimension,
            device=encoder_conditioning.device,
            dtype=dtype,
        )
        if temperature != 1.0:
            x = x * temperature**0.5

        timesteps = self.shift_timesteps(
            torch.linspace(0.0, 1.0, num_sampling_steps + 1, device=encoder_conditioning.device), sequence_length
        )
        null_ssl_conditioning = None if ssl_conditioning is None else torch.zeros_like(ssl_conditioning)
        for step in range(num_sampling_steps):
            t = timesteps[step].expand(batch_size)
            dt = timesteps[step + 1] - timesteps[step]
            velocity = self(x, t, encoder_conditioning, ssl_conditioning)

            if guidance_scale != 1.0 and guidance_target == "ssl":
                weak_velocity = self(x, t, encoder_conditioning, null_ssl_conditioning)
            elif guidance_scale != 1.0:
                weak_velocity = self(x, t, torch.zeros_like(encoder_conditioning), null_ssl_conditioning)

            if guidance_scale != 1.0:
                velocity = weak_velocity + guidance_scale * (velocity - weak_velocity)
            x = x + velocity * dt

            if callback is not None:
                t_next = timesteps[step + 1]
                callback({"i": step, "t": t_next, "x": x, "denoised": x + (1.0 - t_next) * velocity})

        return x

    @torch.inference_mode()
    def restore_audio(
        self,
        noisy_audio: torch.Tensor,
        num_sampling_steps: int | None = None,
        guidance_scale: float | None = None,
        temperature: float = 1.0,
        guidance_target: Literal["all", "ssl"] = "all",
        callback: Callable[[dict], None] | None = None,
    ) -> torch.Tensor:
        """
        Restore degraded audio.

        :param noisy_audio: Degraded audio with shape [B, 1, num_samples].
        :type noisy_audio: torch.Tensor
        :param num_sampling_steps: Number of Euler steps. Defaults to `self.num_sampling_steps`.
        :type num_sampling_steps: int | None
        :param guidance_scale: Guidance scale, defaults to `self.guidance_scale`.
        :type guidance_scale: float | None
        :param temperature: Temperature control from CALM [1].
            The initial noise is drawn with variance `temperature`, i.e. standard deviation `sqrt(temperature)`,
            trading diversity for fidelity below 1.0. 1.0 is plain sampling.
        :type temperature: float
        :param guidance_target: What to take away for the weak branch of classifier-free guidance (CFG).
            `all` zeroes every stream, obtaining the unconditional velocity.
            `ssl` keeps the encoder's features and takes away only the SSL stream.
        :type guidance_target: Literal["all", "ssl"]
        :param callback: Euler step with a dict holding information about the step.
        :type callback: Callable[[dict], None] | None

        :returns: Restored audio with shape [B, 1, num_samples].
        :rtype: torch.Tensor

        .. [1] https://arxiv.org/abs/2509.06926
        """

        _batch_size, _num_channels, num_samples = noisy_audio.shape

        encoder_conditioning, ssl_conditioning = self.get_conditioning(noisy_audio)
        latents = self.sample(
            encoder_conditioning,
            num_sampling_steps,
            guidance_scale,
            ssl_conditioning,
            temperature,
            guidance_target=guidance_target,
            callback=callback,
        )
        return self.decode(latents)[..., :num_samples]
