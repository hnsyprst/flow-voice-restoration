import logging

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from rich.abc import RichRenderable
from rich.logging import RichHandler


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
