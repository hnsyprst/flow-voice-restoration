from collections.abc import Callable, Iterable
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F  # noqa: N812
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerConfig, OptimizerLRSchedulerConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from flow_voice_restoration.losses.distillation import DistillationLoss
from flow_voice_restoration.models.components.voice_restoration import VoiceRestorer


def source_metric_names(
    metrics: dict[str, torch.Tensor],
    source: str,
    is_reference: bool,
) -> dict[str, torch.Tensor]:
    """
    Add the dataset source name to a dictionary of validation metrics.

    :param metrics: Dictionary of validation metrics.
    :type metrics: dict[str, torch.Tensor]
    :param source: Dataset source name.
    :type source: str
    :param is_reference: If `True`, don't append the dataset source name.
    :type is_reference: bool

    :returns: Dictionary of validation metrics, with source name added to the keys.
    :rtype: dict[str, torch.Tensor]
    """

    if is_reference:
        return metrics
    return {f"valid/{source}/{name.removeprefix('valid/')}": value for name, value in metrics.items()}


def sum_loss_terms(
    loss: dict[str, torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """
    Sum loss terms.

    :param loss: Dictionary of loss terms, or a tensor containing a single loss term.
    :type loss: dict[str, torch.Tensor] | torch.Tensor

    :returns: Summed loss, as a tensor.
    :rtype: torch.Tensor
    """

    if isinstance(loss, torch.Tensor):
        return loss
    if not loss:
        return torch.zeros(())
    return torch.stack(list(loss.values())).sum()


class OptimizerProtocol(Protocol):
    """
    Protocol defining the interface for optimizers used to train the voice restoration model.
    """

    keywords: dict[str, Any]

    def __call__(self, params: Iterable[torch.Tensor] | Iterable[dict[str, Any]]) -> Optimizer: ...


class VoiceRestorationLightningModule(LightningModule):
    """
    Lightning module for the flow-matching voice restoration model.
    """

    def __init__(
        self,
        net: VoiceRestorer,
        optimizer: OptimizerProtocol,
        scheduler: Callable[..., LRScheduler] | None = None,
        encoder_lr_scale: float = 1.0,
        conditioning_dropout_probability: float = 0.1,
        ssl_dropout_probability: float = 0.0,
        timestep_mean: float = 0.0,
        timestep_std: float = 1.0,
        audio_reconstruction_criterion: torch.nn.Module | None = None,
        audio_reconstruction_criterion_weight: float = 0.0,
        content_distillation_criterion: DistillationLoss | None = None,
        content_distillation_criterion_weight: float = 0.0,
        num_validation_sampling_steps: int = 8,
        strict_loading: bool = True,
    ):
        """
        Initialise the lightning module.

        :param net: The flow-matching voice restoration model to train.
        :type net: VoiceRestorer
        :param optimizer: The optimizer class to use for training.
        :type optimizer: OptimizerProtocol
        :param scheduler: The learning rate scheduler class to use for training, or `None` to disable scheduling.
        :type scheduler: Callable[..., LRScheduler] | None
        :param encoder_lr_scale: Scale factor for the learning rate of the encoder parameters.
        :type encoder_lr_scale: float
        :param conditioning_dropout_probability: Probability of dropping the conditioning during training,
            to enable classifier-free guidance at inference.
        :type conditioning_dropout_probability: float
        :param ssl_dropout_probability: Probability of dropping the SSL conditioning during training,
            to enable classifier-free guidance at inference.
        :type ssl_dropout_probability: float
        :param timestep_mean: Mean of the Gaussian distribution used to sample timesteps for flow matching.
        :type timestep_mean: float
        :param timestep_std: Standard deviation of the Gaussian distribution used to sample timesteps for flow matching.
        :type timestep_std: float
        :param audio_reconstruction_criterion: Criterion for audio reconstruction loss, or `None` to disable it.
        :type audio_reconstruction_criterion: torch.nn.Module | None
        :param audio_reconstruction_criterion_weight: Weight of the audio reconstruction loss in the total loss.
        :type audio_reconstruction_criterion_weight: float
        :param content_distillation_criterion: Criterion for content distillation loss, or `None` to disable it.
        :type content_distillation_criterion: DistillationLoss | None
        :param num_validation_sampling_steps: Number of Euler steps to take when computing metrics during validation.
        :type num_validation_sampling_steps: int
        :param strict_loading: Enables or disables strict loading of checkpoints.
        :type strict_loading: bool
        """

        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "audio_reconstruction_criterion",
                "content_distillation_criterion",
            ]
        )

        if not 0.0 <= ssl_dropout_probability < 1.0:
            raise ValueError(f"`ssl_dropout_probability` must be in [0, 1), got {ssl_dropout_probability}.")
        if ssl_dropout_probability > 0.0 and net.ssl_conditioner is None:
            raise ValueError("`ssl_dropout_probability` needs a net with an `ssl_conditioner`; this one has none.")

        self.net = net

        self.optimizer_class = optimizer
        self.scheduler_class = scheduler
        self.encoder_lr_scale = encoder_lr_scale

        self.conditioning_dropout_probability = conditioning_dropout_probability
        self.ssl_dropout_probability = ssl_dropout_probability

        self.timestep_mean = timestep_mean
        self.timestep_std = timestep_std
        self.num_validation_sampling_steps = num_validation_sampling_steps

        self.audio_reconstruction_criterion = audio_reconstruction_criterion
        self.audio_reconstruction_criterion_weight = audio_reconstruction_criterion_weight
        self.content_distillation_criterion = content_distillation_criterion
        self.content_distillation_criterion_weight = content_distillation_criterion_weight

        self.strict_loading = strict_loading

    def _flow_step(
        self,
        noisy_audio: torch.Tensor,
        clean_latents: torch.Tensor,
        drop_conditioning: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None]]:
        """
        Calculate the flow matching loss on a batch.

        :param noisy_audio: Degraded audio with shape [B, 1, num_samples].
        :type noisy_audio: torch.Tensor
        :param clean_latents: Clean standardized latents with shape [B, T, latent_dimension].
        :type clean_latents: torch.Tensor
        :param drop_conditioning: Whether to zero the conditioning of a random subset of the batch,
            to enable classifier-free guidance at inference.
        :type drop_conditioning: bool

        :returns: The velocity loss and the conditioning streams *before* dropout,
            for callers that want the conditional path.
        :rtype: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None]]
        """

        encoder_conditioning, ssl_conditioning = self.net.get_conditioning(noisy_audio)
        num_frames = min(encoder_conditioning.shape[1], clean_latents.shape[1])
        encoder_conditioning = encoder_conditioning[:, :num_frames]
        if ssl_conditioning is not None:
            ssl_conditioning = ssl_conditioning[:, :num_frames]
        target_latents = clean_latents[:, :num_frames]
        batch_size = target_latents.shape[0]

        model_encoder_conditioning = encoder_conditioning
        model_ssl_conditioning = ssl_conditioning
        if drop_conditioning and self.conditioning_dropout_probability > 0.0:
            dropout_mask = (
                torch.rand(batch_size, 1, 1, device=encoder_conditioning.device) < self.conditioning_dropout_probability
            )
            model_encoder_conditioning = torch.where(
                dropout_mask, torch.zeros_like(encoder_conditioning), encoder_conditioning
            )
            # The same samples lose both streams. Dropping one and keeping the other would teach the
            # unconditional branch to lean on whichever survived, and classifier-free guidance would
            # then be extrapolating along the wrong axis
            if ssl_conditioning is not None:
                model_ssl_conditioning = torch.where(dropout_mask, torch.zeros_like(ssl_conditioning), ssl_conditioning)
        if drop_conditioning and self.ssl_dropout_probability > 0.0 and model_ssl_conditioning is not None:
            ssl_dropout_mask = (
                torch.rand(batch_size, 1, 1, device=encoder_conditioning.device) < self.ssl_dropout_probability
            )
            model_ssl_conditioning = torch.where(
                ssl_dropout_mask, torch.zeros_like(model_ssl_conditioning), model_ssl_conditioning
            )

        timestep = torch.sigmoid(
            torch.randn(batch_size, device=target_latents.device) * self.timestep_std + self.timestep_mean
        )
        timestep = self.net.shift_timesteps(timestep, num_frames)
        noise = torch.randn_like(target_latents)
        t = timestep[:, None, None]
        interpolant = (1.0 - t) * noise + t * target_latents
        target_velocity = target_latents - noise

        predicted_velocity = self.net(interpolant, timestep, model_encoder_conditioning, model_ssl_conditioning)
        velocity_loss = F.mse_loss(predicted_velocity, target_velocity)

        return velocity_loss, (encoder_conditioning, ssl_conditioning)

    def training_step(
        self,
        batch,
        batch_idx: int,
    ) -> torch.Tensor:
        noisy_audio, clean_audio, clean_latents, *content_features = batch

        velocity_loss, conditioning = self._flow_step(noisy_audio, clean_latents, drop_conditioning=True)
        content_distillation_loss = torch.zeros((), device=clean_audio.device)
        if self.content_distillation_criterion is not None:
            content_distillation_loss = (
                self.content_distillation_criterion(conditioning[0], content_features[0])
                * self.content_distillation_criterion_weight
            )
        total_loss = velocity_loss + content_distillation_loss

        self.log_dict(
            {
                "train/velocity_loss": velocity_loss.detach(),
                "train/content_distillation_loss": content_distillation_loss.detach(),
                "train/loss": total_loss.detach(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return total_loss

    def validation_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        noisy_audio, clean_audio, clean_latents, *content_features = batch
        # Set only under a datamodule with several named sources (`MultiSourceDataModule`);
        # with any other, validation logs exactly what it always has
        source_names = getattr(getattr(self._trainer, "datamodule", None), "source_names", None)

        velocity_loss, flow_step_conditioning = self._flow_step(noisy_audio, clean_latents, drop_conditioning=False)
        encoder_conditioning, ssl_conditioning = flow_step_conditioning
        sampled_latents = self.net.sample(
            encoder_conditioning,
            num_sampling_steps=self.num_validation_sampling_steps,
            ssl_conditioning=ssl_conditioning,
        )
        latent_distance_loss = F.mse_loss(sampled_latents, clean_latents[:, : encoder_conditioning.shape[1]])

        total_loss = latent_distance_loss
        metrics = {
            "valid/velocity_loss": velocity_loss.detach(),
            "valid/latent_distance_loss": latent_distance_loss.detach(),
        }

        if self.content_distillation_criterion is not None:
            metrics["valid/content_distillation_loss"] = (
                self.content_distillation_criterion(flow_step_conditioning[0], content_features[0])
                * self.content_distillation_criterion_weight
            ).detach()
        if self.audio_reconstruction_criterion is not None:
            restored_audio = self.net.decode(sampled_latents)
            num_samples = min(restored_audio.shape[-1], clean_audio.shape[-1])

            if self.audio_reconstruction_criterion is not None:
                audio_reconstruction_loss = (
                    sum_loss_terms(
                        self.audio_reconstruction_criterion(
                            restored_audio[..., :num_samples], clean_audio[..., :num_samples]
                        )
                    )
                    * self.audio_reconstruction_criterion_weight
                )
                total_loss = total_loss + audio_reconstruction_loss
                metrics["valid/audio_reconstruction_loss"] = audio_reconstruction_loss.detach()

            if source_names is not None:
                rms = restored_audio[..., :num_samples].float().pow(2).mean(dim=(-2, -1)).sqrt()
                metrics["valid/estimate_rms_dbfs"] = (20.0 * torch.log10(rms + 1e-8)).mean().detach()

        metrics["valid/loss"] = total_loss.detach()
        if source_names is None:
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            return
        is_reference = dataloader_idx == 0
        self.log_dict(
            source_metric_names(metrics, source_names[dataloader_idx], is_reference),
            on_step=False,
            on_epoch=True,
            prog_bar=is_reference,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def configure_optimizers(
        self,
    ) -> OptimizerConfig | OptimizerLRSchedulerConfig:
        base = getattr(self.net, "_orig_mod", self.net)
        weight_decay = self.optimizer_class.keywords.get("weight_decay", 0.0)
        base_lr = self.optimizer_class.keywords["lr"]
        encoder_lr = base_lr * self.encoder_lr_scale
        encoder_param_ids = {id(param) for param in base.encoder.parameters()}

        groups: dict[str, list[torch.nn.Parameter]] = {
            "encoder_decay": [],
            "encoder_no_decay": [],
            "velocity_decay": [],
            "velocity_no_decay": [],
        }
        for name, param in base.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in encoder_param_ids:
                suffix = "decay" if "transformer" in name else "no_decay"
                groups[f"encoder_{suffix}"].append(param)
            else:
                suffix = "decay" if param.ndim >= 2 else "no_decay"
                groups[f"velocity_{suffix}"].append(param)

        distillation_params: list[torch.nn.Parameter] = []
        if self.content_distillation_criterion is not None:
            distillation_params = [
                param for param in self.content_distillation_criterion.parameters() if param.requires_grad
            ]

        param_groups = [
            {"params": groups["encoder_decay"], "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": groups["encoder_no_decay"], "lr": encoder_lr, "weight_decay": 0.0},
            {"params": groups["velocity_decay"], "lr": base_lr, "weight_decay": weight_decay},
            {"params": groups["velocity_no_decay"], "lr": base_lr, "weight_decay": 0.0},
            {"params": [p for p in distillation_params if p.ndim >= 2], "lr": base_lr, "weight_decay": weight_decay},
            {"params": [p for p in distillation_params if p.ndim < 2], "lr": base_lr, "weight_decay": 0.0},
        ]
        param_groups = [group for group in param_groups if group["params"]]
        optimizer = self.optimizer_class(params=param_groups)

        if self.scheduler_class is None:
            return cast(OptimizerConfig, {"optimizer": optimizer})

        scheduler = self.scheduler_class(optimizer=optimizer, total_steps=int(self.trainer.estimated_stepping_batches))
        return cast(
            OptimizerLRSchedulerConfig,
            {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            },
        )
