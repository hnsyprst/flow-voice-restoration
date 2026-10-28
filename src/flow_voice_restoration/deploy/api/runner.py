import math
import struct
from functools import partial
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

import torch
from lauryn.errors import RunnerError
from lauryn.runners import TorchBaseRunner
from lauryn.structs import CheckpointLoader

from flow_voice_restoration.inference.restoration import count_windows, restore
from flow_voice_restoration.inference.utils import load_config, load_net
from flow_voice_restoration.models.components.voice_restoration import VoiceRestorer
from flow_voice_restoration.utils.rich_logging import TensorProgress


def restorer_loader(
    checkpoint_path: Path,
    experiment: str,
) -> CheckpointLoader:
    """
    Make a loader for a restorer checkpoint, composing its experiment's config now.

    :param checkpoint_path: Lightning checkpoint holding the `net.*` weights.
    :type checkpoint_path: Path
    :param experiment: Experiment config the checkpoint was trained with.
    :type experiment: str

    :returns: The loader, for `VoiceRestorerRunner`'s `model_loaders["restorer"]`.
    :rtype: CheckpointLoader
    """

    return CheckpointLoader(str(checkpoint_path), partial(load_net, load_config(experiment)))


class VoiceRestorerRunner(TorchBaseRunner):
    """
    Runner for the voice restoration model.
    """

    def __init__(
        self,
        model_loaders: dict[str, CheckpointLoader],
        device: str = "cpu",
        window_seconds: float = 10.0,
        overlap_seconds: float = 2.0,
        batch_size: int = 4,
        input_peak: float = 0.9,
        preview_rows: int = 256,
        preview_columns: int = 256,
        preview_queue: Queue | None = None,
    ):
        """
        Initialise the runner.

        :param model_loaders: Loaders for the models to run, with the restorer under the key "restorer".
        :type model_loaders: dict[str, CheckpointLoader]
        :param device: Device to run the models on.
        :type device: str
        :param window_seconds: Window length in seconds, snapped to the encoder's frame hop.
        :type window_seconds: float
        :param overlap_seconds: Overlap between windows in seconds, snapped to the encoder's frame hop.
            At most half of `window_seconds`.
        :type overlap_seconds: float
        :param batch_size: How many windows to restore per forward pass.
        :type batch_size: int
        :param input_peak: The peak inputs are normalised to during training.
        :type input_peak: float
        :param preview_rows: Most rows a preview's grid holds.
        :type preview_rows: int
        :param preview_columns: Most columns a preview's grid holds.
        :type preview_columns: int
        :param preview_queue: Queue to put latent on while restoring audio.
        :type preview_queue: Queue | None
        """

        super().__init__(
            model_loaders=model_loaders,
            device=device,
        )
        net = self.models["restorer"]
        if not isinstance(net, VoiceRestorer):
            raise ValueError(f"Expected to load a `VoiceRestorer` model, but got {type(net)}")

        frame_hop = round(net.sample_rate / net.encoder_frame_rate)
        self.window = max(1, round(window_seconds * net.sample_rate / frame_hop)) * frame_hop
        self.overlap = round(overlap_seconds * net.sample_rate / frame_hop) * frame_hop
        if 2 * self.overlap > self.window:
            raise ValueError(f"Overlap ({overlap_seconds} s) must be at most half of the window ({window_seconds} s).")
        self.batch_size = batch_size
        self.input_peak = input_peak
        self.preview_rows = preview_rows
        self.preview_columns = preview_columns
        self.preview_queue = preview_queue

    def send_preview(
        self,
        preview_id: str,
        denoised: torch.Tensor,
        t: torch.Tensor,
        step: int,
        total: int,
    ):
        """
        Send the first window's current estimate of the denoised latents to `self.preview_queue`.
        Does nothing if `self.preview_queue` is None.

        :param preview_id: The job's preview ID.
        :type preview_id: str
        :param denoised: Denoised latents with shape [B, T, C].
        :type denoised: torch.Tensor
        :param t: Flow time reached by the step just taken.
        :type t: torch.Tensor
        :param step: Steps taken so far, across all batches of windows.
        :type step: int
        :param total: Steps the job takes in total.
        :type total: int
        """

        if self.preview_queue is None:
            return
        grid = TensorProgress.to_grid(denoised.transpose(1, 2), self.preview_columns, self.preview_rows)
        header = struct.pack("<IIfHH", step, total, float(t.reshape(-1)[0]), *grid.shape)
        self.preview_queue.put((preview_id, header + (grid * 255).round().to(torch.uint8).numpy().tobytes()))

    def run(
        self,
        input_data: dict[str, torch.Tensor],
        params: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """
        Restore an input, reporting progress (and previews, given a `preview_id`) after every sampling step.

        :param input_data: The input under `"audio"`, from the mono formatter, with shape [1, num_samples].
        :type input_data: dict[str, torch.Tensor]
        :param params: `num_sampling_steps` and `guidance_scale` (None to use the net's own),
            `guidance_target`, `temperature` and `seed` (or None), and optionally `preview_id`.
        :type params: dict[str, Any]

        :returns: The restored audio under `"audio_out"`, with shape [1, num_samples].
        :rtype: dict[str, torch.Tensor]
        """

        net = self.models["restorer"]
        if not isinstance(net, VoiceRestorer):
            raise ValueError(f"Expected `self.net` to be a `VoiceRestorer`, but got {type(net)}")

        audio = input_data["audio"][0]  # mono formatter gives [1, T]
        peak = audio.abs().max()
        if peak < 1e-4:
            raise RunnerError(502, "input is (near-)silent")
        audio = audio / peak * self.input_peak

        steps = params["num_sampling_steps"] or net.num_sampling_steps
        guidance = params["guidance_scale"] if params["guidance_scale"] is not None else net.guidance_scale
        if params["guidance_target"] == "ssl":
            if net.ssl_conditioner is None:
                raise RunnerError(502, "guidance_target=ssl needs a model with an SSL conditioner")
            if guidance == 1.0:
                raise RunnerError(502, "guidance_target=ssl does nothing at a guidance scale of 1.0")
        if params["seed"] is not None:
            torch.manual_seed(params["seed"])

        total = math.ceil(count_windows(audio.shape[-1], self.window, self.overlap) / self.batch_size) * steps
        preview_id = params.get("preview_id")

        def callback(batch_index: int, d: dict):
            step = batch_index * steps + d["i"] + 1
            self.set_progress(step / total)
            if preview_id is not None:
                self.send_preview(preview_id, d["denoised"], d["t"], step, total)

        restored = restore(
            net,
            audio,
            self.window,
            self.overlap,
            self.batch_size,
            self.device,
            {
                "num_sampling_steps": steps,
                "guidance_scale": guidance,
                "temperature": params["temperature"],
                "guidance_target": params["guidance_target"],
            },
            callback=callback,
        )
        return {"audio_out": restored.clamp(-1.0, 1.0).unsqueeze(0)}
