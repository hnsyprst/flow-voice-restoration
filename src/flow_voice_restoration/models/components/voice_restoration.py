import math
from typing import Literal, Self, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from flow_voice_restoration.models.components.modules.dit import DiT
from flow_voice_restoration.models.components.modules.ssl_conditioning import DenoisedSslConditioner
from flow_voice_restoration.utils.autoencoder import ExportedAutoencoder


class VoiceRestorer(nn.Module):
    """
    Flow-matching voice restoration model.

    A trainable conditioning encoder (warm-started from the frozen stage 1 VAE's encoder) extracts
    features from degraded audio; a `DiT` velocity model then transports Gaussian noise to the
    clean standardized bottleneck latents, conditioned on those features. The frozen VAE bottleneck
    and decoder render the result. Unlike the regression predictor, sampling picks a plausible
    clean latent instead of averaging over all of them, so it does not over-smooth under ambiguity.

    An optional `ssl_conditioner` adds a second view of the degraded audio to that conditioning: the
    denoised w2v-BERT features of `DenoisedSslConditioner`, which describe linguistic content rather
    than signal. Heavy distortion is where the waveform encoder loses the words while keeping pitch,
    timbre and prosody, and where a content-bearing feature stream has something to add. Leave it
    `None` for the model exactly as it was before this existed, parameter for parameter.

    `ssl_conditioning_mode` decides how that second view reaches the velocity model:

      - `add`: pooled to the latent frame rate, projected to the conditioning width and summed into
        the encoder's features, so the velocity model sees one stream. The projection is
        zero-initialised, which is what lets a checkpoint from before the conditioner existed load
        into a model with one.
      - `separate`: handed to the velocity model as its own stream, at the student's 50 Hz and full
        width, for a `DiT` built with an `ssl_conditioning_dimension`. Nothing is pooled or squeezed,
        and the two streams are no longer superimposed -- summing them constrains the velocity
        model's view of the SSL features to a rank-`conditioning_dimension` map of the encoder's own
        projection, where two streams let it read each on its own terms.

    `timestep_shift_base_dimension` turns on RAE's dimension-dependent schedule shift (Zheng et al.,
    "Diffusion Transformers with Representation Autoencoders", §3.2, after Esser et al. 2024). The
    more numbers a sample holds, the less the same noise level corrupts it, so flow time is shifted
    towards noise by alpha = sqrt(m / n), where m is the effective data dimension -- frames times
    latent channels -- and n the base dimension (4096 in both papers). See `shift_timesteps`. Both
    training (via the Lightning module) and sampling go through it, so a model trained shifted is
    sampled shifted. `None` leaves flow time as it always was.

    Sampling can be guided in one of two ways. Classifier-free guidance extrapolates away from the
    velocity model's own unconditional prediction (zeroed conditioning). Autoguidance (Karras et al.,
    "Guiding a Diffusion Model with a Bad Version of Itself", 2024) instead extrapolates away from a
    `guide`: a weaker model -- smaller, or less trained, such as an earlier checkpoint of the same run
    -- trained on the same task and conditioned on the same input. The paper's argument is that the
    difference between the two isolates the errors the weaker model makes more of, so extrapolating
    away from it corrects those without the loss of variation CFG's conditional/unconditional
    mismatch brings. The velocity is affine in the denoiser output at fixed x and t, so guiding
    velocities is the same as guiding denoisers.

    CFG can also take away the SSL stream alone (`guidance_target="ssl"`), extrapolating from the
    encoder-only prediction towards the full one: v = v(enc) + w * (v(enc + ssl) - v(enc)). That
    pushes along the content direction only, not timbre, noise and room as well. It needs a model
    trained with the SSL stream dropped on its own (the Lightning module's `ssl_dropout_probability`);
    without that, the encoder-only branch is a condition the model has never seen.

    The stage 1 model is a SAME autoencoder (see `load_exported_autoencoder`). Its
    decoder rounds the output length up past the input's; trim to the input length where it matters.
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

        :param autoencoder: Autoencoder used to decode predicted latents, with its sample rate (see
            `load_exported_autoencoder`). Its encoder is also used to warm-start the (optionally
            trainable) conditioning encoder. The bottleneck and decoder are frozen during training.
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
        :param timestep_shift_base_dimension: Base dimension n of the dimension-dependent timestep shift,
            set to 4096 by default in the RAE recipe. `None` disables the shift.
        :type timestep_shift_base_dimension: int | None
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
        placeholder
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
        Apply the dimension-dependent timestep shift, or return `timestep` unchanged if it is off.

        RAE states the shift with time running the other way (t=1 is noise): s' = alpha * s /
        (1 + (alpha - 1) * s). Here t=0 is noise, so s = 1 - t, which gives t' = t / (1 + (alpha - 1)
        * (1 - t)). Both ends stay fixed, and with alpha > 1 everything in between moves towards noise.

        :param timestep: Flow times in [0, 1], any shape.
        :type timestep: torch.Tensor
        :param num_frames: Latent frames per sample. Times the latent dimension, this is the effective
            data dimension m in alpha = sqrt(m / n).
        :type num_frames: int

        :returns: Shifted flow times, same shape as `timestep`.
        :rtype: torch.Tensor
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
    ) -> torch.Tensor:
        """
        Sample clean standardized latents by Euler-integrating the learned velocity field
        from Gaussian noise (t=0) to data (t=1).

        Every step stays on the autograd graph, so a loss on the result reaches the velocity model
        through all `num_sampling_steps` of it. That graph costs one velocity-model activation set
        per step; prefer `sample` unless you specifically need the gradients.

        :param encoder_conditioning: The encoder's conditioning features with shape
            [B, T, conditioning_dimension].
        :type encoder_conditioning: torch.Tensor
        :param num_sampling_steps: Number of Euler steps. Defaults to `self.num_sampling_steps`.
        :type num_sampling_steps: int | None
        :param guidance_scale: Guidance scale -- CFG's, or autoguidance's when `guide` is given. 1.0
            disables either. Defaults to `self.guidance_scale`.
        :type guidance_scale: float | None
        :param ssl_conditioning: The SSL stream with shape [B, T, conditioning_dimension], summed into
            `encoder_conditioning` by `forward`, or `None` for a model without an `ssl_conditioner`.
            Plain CFG drops it alongside the encoder's conditioning, so the unconditional branch is
            genuinely unconditional rather than still holding the content.
        :type ssl_conditioning: torch.Tensor | None
        :param temperature: Gaussian temperature (Rouard et al. 2026, CALM): the initial noise is
            drawn with variance `temperature`, i.e. standard deviation `sqrt(temperature)`, trading
            diversity for fidelity below 1.0. 1.0 is plain sampling.
        :type temperature: float
        :param guidance_target: What CFG takes away for its weak branch. `all` zeroes every stream, as
            CFG always has. `ssl` keeps the encoder's features and takes away only the SSL stream, so
            the guidance pushes along the content direction alone (see this class's docstring). Not
            combinable with `guide`, whose weak branch is a different model.
        :type guidance_target: Literal["all", "ssl"]

        :returns: Sampled latents with shape [B, T, latent_dimension].
        :rtype: torch.Tensor
        """

        num_sampling_steps = num_sampling_steps or self.num_sampling_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.guidance_scale

        if guidance_target not in ["all", "ssl"]:
            raise ValueError(f"`guidance_target` must be one of ['all', 'ssl'], got {guidance_target!r}.")
        if guidance_target == "ssl" and guidance_scale != 1.0 and ssl_conditioning is None:
            raise ValueError("SSL-only guidance needs `ssl_conditioning`; none was given.")

        # The noise is drawn in the dtype of the summed conditioning `forward` hands the velocity
        # model, which is what it was drawn in when the streams were summed before sampling.
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

        return x

    @torch.inference_mode()
    def restore_audio(
        self,
        noisy_audio: torch.Tensor,
        num_sampling_steps: int | None = None,
        guidance_scale: float | None = None,
        temperature: float = 1.0,
        guidance_target: Literal["all", "ssl"] = "all",
    ) -> torch.Tensor:
        """
        Restore degraded audio.

        :param noisy_audio: Degraded audio with shape [B, 1, num_samples].
        :type noisy_audio: torch.Tensor
        :param num_sampling_steps: Number of Euler steps. Defaults to `self.num_sampling_steps`.
        :type num_sampling_steps: int | None
        :param guidance_scale: Guidance scale -- CFG's, or autoguidance's when `guide` is given.
            Defaults to `self.guidance_scale`.
        :type guidance_scale: float | None
        :param temperature: Gaussian temperature of the initial noise; see `integrate`.
        :type temperature: float
        :param guidance_target: What CFG takes away for its weak branch, `all` or `ssl`; see `integrate`.
        :type guidance_target: Literal["all", "ssl"]

        :returns: Restored audio with shape [B, 1, num_samples].
        :rtype: torch.Tensor
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
        )
        return self.decode(latents)[..., :num_samples]
