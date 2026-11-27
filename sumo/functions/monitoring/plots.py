"""Static report figures; rendered by the background monitoring worker."""

import os
import threading
from functools import wraps
from itertools import groupby
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

_plot_lock = threading.RLock()


def serialized_plot(function):
    """Matplotlib shares internal state even when using separate Figure objects."""

    @wraps(function)
    def render(*args, **kwargs):
        with _plot_lock:
            return function(*args, **kwargs)

    return render


@serialized_plot
def performance_figure(data, path):
    stages = data["stages"]
    if not stages:
        return
    height = max(6.5, 3.5 + 0.29 * len(stages))
    fig = Figure(figsize=(14, height), facecolor="#f6f8fc", layout="constrained")
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, 2, height_ratios=[max(2, len(stages) * 0.23), 1.5])
    duration = fig.add_subplot(grid[0, 0])
    memory = fig.add_subplot(grid[0, 1])
    timeline = fig.add_subplot(grid[1, :])
    labels = [
        row["stage"].removeprefix("sumo.")
        + (" [retained]" if not row.get("current_run", True) else "")
        for row in stages
    ]
    palette = {
        "This run": "#137c8b",
        "Retained": "#7b879b",
        "Failed": "#b54747",
        "Interrupted": "#d77731",
        "Running": "#edae49",
    }
    kinds = [
        row["status"].capitalize()
        if row["status"] != "completed"
        else "This run"
        if row.get("current_run", True)
        else "Retained"
        for row in stages
    ]
    colors = [palette.get(kind, "#7b879b") for kind in kinds]
    for axis, values, title in (
        (
            duration,
            [(row["seconds"] or 0) / 60 for row in stages],
            "Wall time · minutes",
        ),
        (
            memory,
            [row["peak_rss_gib"] for row in stages],
            "Peak process-tree RSS · GiB",
        ),
    ):
        bars = axis.barh(range(len(labels)), values, color=colors, height=0.65)
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        axis.set_title(title, loc="left", weight="bold", pad=14)
        axis.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
        axis.set_xlim(0, max(values + [0.1]) * 1.18)
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)
    memory.tick_params(axis="y", labelleft=False)
    for index, segment in groupby(
        data["samples"], key=lambda row: row.get("stage_index", 0)
    ):
        points = list(segment)
        x = [row["seconds"] / 60 for row in points]
        y = [row["rss_gib"] for row in points]
        color = colors[index]
        timeline.fill_between(x, y, color=color, alpha=0.15)
        timeline.plot(x, y, color=color, linewidth=1.5)
    timeline.set(
        xlabel="Combined stage time · minutes (downtime excluded)",
        ylabel="RSS · GiB",
        title="Memory across the latest stage measurements",
    )
    timeline.legend(
        handles=[
            Patch(color=palette[kind], label=kind)
            for kind in dict.fromkeys(kinds)
            if kind in palette
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    timeline.grid(alpha=0.15)
    for axis in (duration, memory, timeline):
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    fig.suptitle(
        "SUMO | Runtime & memory", fontsize=21, weight="bold", x=0.02, ha="left"
    )
    job = data.get("slurm_job_id") or "local"
    threads = data.get("allocated_threads") or "not reported"
    retained = sum(not row.get("current_run", True) for row in stages)
    fig.supxlabel(
        f"Job {job} · allocated CPUs: {threads} · {retained} retained / {len(stages) - retained} measured this run · sampled RSS includes child processes; shared pages may repeat",
        fontsize=8,
    )
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    fig.savefig(temporary, format="png", dpi=150)
    os.replace(temporary, path)
    fig.clear()
