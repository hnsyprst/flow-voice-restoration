import argparse
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import torch
from fastapi import BackgroundTasks, File, Form, UploadFile
from lauryn.base import BaseAPI
from lauryn.formatters import AUDIO_FORMATTER_44KHZ_MONO
from lauryn.structs import AUDIO_FORMAT
from lauryn.validators import AUDIO_VALIDATOR_44KHZ_MONO_5MIN

from flow_voice_restoration.deploy.api.previews import PreviewBroadcaster
from flow_voice_restoration.deploy.api.runner import VoiceRestorerRunner, restorer_loader

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the voice restoration model.")
    parser.add_argument(
        "experiment",
        help="Experiment config the checkpoint was trained with",
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Checkpoint to restore with",
    )
    parser.add_argument(
        "--window-length",
        type=float,
        default=10.0,
        help="Window length in seconds",
    )
    parser.add_argument(
        "--overlap-length",
        type=float,
        default=2.0,
        help="Overlap between windows in seconds",
    )
    parser.add_argument(
        "-b",
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
        "--num-workers",
        type=int,
        default=0,
        help="Number of GPU worker processes",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    parser.add_argument(
        "--previews",
        action="store_true",
        help="Stream sampler previews over the /preview/{preview_id} WebSocket",
    )
    args = parser.parse_args()

    broadcaster = PreviewBroadcaster() if args.previews else None
    runner = VoiceRestorerRunner(
        model_loaders={"restorer": restorer_loader(args.checkpoint, args.experiment)},
        device=args.device,
        window_seconds=args.window_length,
        overlap_seconds=args.overlap_length,
        batch_size=args.batch_size,
        preview_queue=None if broadcaster is None else broadcaster.queue,
    )
    app = BaseAPI(
        title="Voice Restoration API",
        runner=runner,
        num_workers=args.num_workers,
        input_formatters={"audio": AUDIO_FORMATTER_44KHZ_MONO},
        input_validators={"audio": AUDIO_VALIDATOR_44KHZ_MONO_5MIN},
        output_validators={"audio_out": AUDIO_VALIDATOR_44KHZ_MONO_5MIN},
        output_formatters={"audio_out": AUDIO_FORMATTER_44KHZ_MONO},
        use_task_queue=True,
    )

    @app.post("/restore", tags=["Submit Task"])
    async def restore(
        background_tasks: BackgroundTasks,
        file: Annotated[UploadFile, File()],
        num_sampling_steps: Annotated[int | None, Form(ge=1)] = None,
        guidance_scale: Annotated[float | None, Form()] = None,
        guidance_target: Annotated[str, Form(pattern="^(all|ssl)$")] = "all",
        temperature: Annotated[float, Form(gt=0.0)] = 1.0,
        seed: Annotated[int | None, Form()] = None,
        out_format: Annotated[AUDIO_FORMAT, Form()] = AUDIO_FORMAT.wav,
    ):
        params = {
            "num_sampling_steps": num_sampling_steps,
            "guidance_scale": guidance_scale,
            "guidance_target": guidance_target,
            "temperature": temperature,
            "seed": seed,
            "audio_format": out_format,
        }
        if broadcaster is None:
            return await app.process({"audio": file}, params, background_tasks)

        preview_id = str(uuid4())
        task = await app.process({"audio": file}, {**params, "preview_id": preview_id}, background_tasks)
        return {**task.model_dump(by_alias=True, mode="json"), "previewId": preview_id}

    if broadcaster is not None:
        app.include_router(broadcaster.router)

    uvicorn.run(app, host=args.host, port=args.port)
