import logging
import time
from itertools import groupby

import torch
import torch.nn.functional as F  # noqa: N812
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from rich.abc import RichRenderable
from rich.color import Color
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.style import Style
from rich.text import Text

RAMPS = {
    "classic": " .:-=+*#%@",
    "blocks": " ░▒▓█",
    "bars": " ▁▂▃▄▅▆▇█",
    "minimal": " .oO@",
}

COLORMAPS = {
    "magma": [(0, 0, 4), (80, 18, 123), (182, 54, 121), (252, 137, 97), (252, 253, 191)],
    "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    "ice": [(4, 8, 24), (30, 64, 140), (80, 160, 220), (190, 230, 250), (255, 255, 255)],
    "ember": [(10, 4, 4), (110, 20, 10), (220, 90, 20), (250, 190, 70), (255, 255, 230)],
}


class RenderableHandler(RichHandler):
    """
    Modifies the RichHandler to pass Renderables straight through without transforming
    them into Text.
    """

    @rank_zero_only
    def emit(self, record):
        super().emit(record)

    def render_message(self, record, message):
        if isinstance(message, RichRenderable):
            return message
        else:
            return super().render_message(record, message)


class RenderableFormatter(logging.Formatter):
    """
    Modify the formatter to pass Renderables straight through without casting them
    to str.
    """

    def format(self, record):
        if isinstance(record.msg, RichRenderable):
            return record.msg
        else:
            return super().format(record)


class TensorProgress:
    """
    Animates the changes in a tensor, alongside a progress bar.
    Fun for watching the noise change along a diffusion/flow-matching reverse process.
    Tensors are drawn channels-first, [B, C, T]; transpose time-major latents before passing them in.
    """

    def __init__(
        self,
        total: int | None = None,
        width: int = 100,
        height: int | None = None,
        ramp: str = "classic",
        color: str | None = "magma",
        fps: float = 12.0,
        title: str = "Generating",
        console: Console | None = None,
        transient: bool = False,
    ):
        """
        Initialise the view.

        :param total: Number of steps that will be taken, if known.
        :type total: int | None
        :param width: Width of the animation in characters. Clamped to fit the console.
        :type width: int
        :param height: Height of the animation in characters. Defaults to filling the console.
        :type height: int
        :param ramp: Name of a ramp in `RAMPS`, or the ramp's characters themselves,
            ordered from quietest to loudest.
        :type ramp: str
        :param color: Name of a colormap in `COLORMAPS`.
            The animation is left uncoloured if omitted.
        :type color: str | None
        :param fps: Most frames to draw per second.
        :type fps: float
        :param title: Title for the panel the animation is drawn in.
        :type title: str
        :param console: Console to draw into. A fresh `Console` is used if omitted.
        :type console: Console | None
        :param transient: Whether to clear the panel when the context exits.
        :type transient: bool
        """

        self.console = console or Console()
        self.ramp = RAMPS.get(ramp, ramp)
        self.styles = (
            [
                Style(color=Color.from_rgb(*self._lerp_rgb(i / (len(self.ramp) - 1), COLORMAPS[color])))
                for i in range(len(self.ramp))
            ]
            if color
            else None
        )
        self.width = max(8, min(width, self.console.width - 4))
        self.height = max(4, min(height or self.console.height - 8, self.console.height - 8))
        self.total = total
        self.fps = fps
        self.title = title
        self.body = Text()
        self.subtitle = ""
        self.drawn_at = 0.0

        self.progress = Progress(
            TextColumn("[bold]step"),
            BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )
        self.task = self.progress.add_task("sampling", total=total)
        self.live = Live(
            self.panel(),
            console=self.console,
            refresh_per_second=fps,
            transient=transient,
            vertical_overflow="crop",
        )

    def _lerp_rgb(
        self,
        v: float,
        stops,
    ) -> tuple:
        """
        Interpolate a colormap between its stops.

        :param v: Position along the colormap, in [0, 1].
        :type v: float
        :param stops: The colormap's RGB stops, evenly spaced along it.
        :type stops: list[tuple[int, int, int]]

        :returns: The colour at `v` as RGB.
        :rtype: tuple
        """

        f = v * (len(stops) - 1)
        i = min(int(f), len(stops) - 2)
        a, b = stops[i], stops[i + 1]
        return tuple(int(round(a[k] + (b[k] - a[k]) * (f - i))) for k in range(3))

    def _to_grid(
        self,
        z: torch.Tensor,
        width: int,
        rows: int,
    ) -> torch.Tensor:
        """
        Reduce a noise tensor to a grid of intensities, one cell per character drawn.
        Channels are pooled to RMS over time, then each channel is normalized to its own
        1st-99th percentile so that quiet channels stay legible.

        :param z: Noise tensor with shape [B, C, T]. Only the first item is drawn.
        :type z: torch.Tensor
        :param width: Most columns the grid may hold.
        :type width: int
        :param rows: Most rows the grid may hold. Channels are pooled together to fit.
        :type rows: int

        :returns: The grid with shape [<=`rows`, <=`width`], in [0, 1].
        :rtype: torch.Tensor
        """

        x = z[:1].detach().float().cpu()
        x = F.adaptive_avg_pool1d(x.pow(2), min(width, x.shape[-1])).sqrt()
        if x.shape[1] > rows:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), rows).transpose(1, 2)
        x = x[0]
        lo = x.quantile(0.01, dim=1, keepdim=True)
        hi = x.quantile(0.99, dim=1, keepdim=True)
        return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)

    def _render(
        self,
        grid: torch.Tensor,
        ramp: str,
        styles: list[Style] | None,
    ) -> Text:
        """
        Draw a grid as ramp characters, one per cell.

        :param grid: Intensities in [0, 1], from `_to_grid()`.
        :type grid: torch.Tensor
        :param ramp: Ramp characters, ordered from quietest to loudest.
        :type ramp: str
        :param styles: One style per ramp character. The text is left uncoloured when omitted.
        :type styles: list | None

        :returns: The drawn grid, one line per row.
        :rtype: Text
        """

        levels = (grid * (len(ramp) - 1)).round().long().tolist()
        text = Text(no_wrap=True)
        for r, row in enumerate(levels):
            if r:
                text.append("\n")
            for level, run in groupby(row):
                text.append(ramp[level] * sum(1 for _ in run), styles[level] if styles else None)
        return text

    def panel(
        self,
    ) -> Panel:
        """
        Returns a panel containing the latest animation frame and the progress bar.

        :returns: The panel the live view draws.
        :rtype: Panel
        """

        return Panel(
            Group(self.body, Text(""), self.progress),
            title=f"[bold]{self.title}[/]",
            subtitle=self.subtitle or None,
            width=self.width + 4,
            padding=(1, 1),
            border_style="grey42",
        )

    def update(
        self,
        z: torch.Tensor,
        index: int | None = None,
        t=None,
    ):
        """
        Advance the progress bar and redraw the animation.
        Frames arriving faster than `fps` are dropped, except for the last step, which is always drawn.

        :param z: Tensor to animate (e.g. the current estimate of the denoised latents), with shape [B, C, T].
        :type z: torch.Tensor
        :param index: Index of the step just taken. The progress bar is left alone when omitted.
        :type index: int | None
        :param t: Flow time reached by the step just taken, shown in the subtitle. Only its first element is read.
        :type t: torch.Tensor | float | None
        """

        if index is not None:
            self.progress.update(self.task, completed=index + 1)
        now = time.monotonic()
        if now - self.drawn_at < 1 / self.fps and index != (self.total or 0) - 1:
            return
        self.drawn_at = now

        self.body = self._render(self._to_grid(z, self.width, self.height), self.ramp, self.styles)
        shape = f"{z.shape[1]}ch × {z.shape[2]}"
        flow_time = "" if t is None else f"t={float(torch.as_tensor(t).reshape(-1)[0]):.3f}  "
        self.subtitle = f"[dim]{flow_time}{shape}[/]"
        self.live.update(self.panel())

    def __enter__(
        self,
    ):
        """
        Start drawing the live view.

        :returns: The view, to be fed with `update()`.
        :rtype: TensorProgress
        """

        self.live.start(refresh=True)
        return self

    def __exit__(
        self,
        *exc,
    ):
        """
        Fill the progress bar, draw a last frame, and stop the live view.

        :param exc: The exception that closed the context, if any. Ignored, so exceptions propagate.
        :type exc: tuple
        """

        if self.total is not None:
            self.progress.update(self.task, completed=self.total)
        self.live.update(self.panel(), refresh=True)
        self.live.stop()
