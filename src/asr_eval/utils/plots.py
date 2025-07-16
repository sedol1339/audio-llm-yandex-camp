import matplotlib.pyplot as plt

from typing import Any


def draw_line_with_ticks(
    x1: float,
    x2: float,
    y: float,
    y_tick_width: float,
    ax: plt.Axes,
    **kwargs: Any,
):
    ax.plot([x1, x2], [y, y], **kwargs,) # type: ignore
    ax.plot([x1, x1], [y - y_tick_width / 2, y + y_tick_width / 2], **kwargs,) # type: ignore
    ax.plot([x2, x2], [y - y_tick_width / 2, y + y_tick_width / 2], **kwargs,) # type: ignore