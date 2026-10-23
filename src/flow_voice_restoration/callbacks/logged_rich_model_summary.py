import logging
from typing import Any

from lightning.pytorch.callbacks.rich_model_summary import RichModelSummary
from rich import get_console
from rich.text import Text

log = logging.getLogger(__name__)


class LoggedRichModelSummary(RichModelSummary):
    """
    Wraps Lightning's native `RichModelSummary` to log the model summary,
    instead of printing it directly to the console.
    """

    @staticmethod
    def summarize(
        summary_data: list[tuple[str, list[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        total_flops: int,
        **summarize_kwargs: Any,
    ):
        console = get_console()
        with console.capture() as capture:
            RichModelSummary.summarize(
                summary_data,
                total_parameters,
                trainable_parameters,
                model_size,
                total_training_modes,
                total_flops,
                **summarize_kwargs,
            )
        log.info(Text.from_ansi(capture.get()))
