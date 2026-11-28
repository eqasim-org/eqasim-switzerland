"""Stream regional SUMO outputs into a compact, inspectable quality report."""

import csv
import json
import math
from collections import Counter
from html import escape
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from scripts.monitoring.plots import serialized_plot
from sumo.functions.scenario.tools import elements


def number(element, name, default=0.0):
    value = float(element.get(name, default))
    return value if math.isfinite(value) else default


def trip_statistics(path):
    """Count all states exactly; duration quantiles use one-second bins."""
    states = Counter()
    classes = Counter()
    totals = Counter()
    histograms = {
        name: Counter()
        for name in ("duration", "waitingTime", "timeLoss", "departDelay")
    }
    for trip in elements(path, ("tripinfo",)):
        depart = number(trip, "depart", -1)
        arrival = number(trip, "arrival", -1)
        state = (
            "undeparted" if depart < 0 else "unfinished" if arrival < 0 else "completed"
        )
        if trip.get("vaporized"):
            state = "vaporized"
        states[state] += 1
        classes[trip.get("vType", "unspecified")] += 1
        if state != "completed":
            continue
        for name, histogram in histograms.items():
            value = max(0.0, number(trip, name))
            totals[name] += value
            histogram[round(value)] += 1
        totals["routeLength"] += number(trip, "routeLength")
    count = states["completed"]
    metrics = {}
    for name, histogram in histograms.items():
        cumulative = 0
        quantiles = {}
        for seconds, frequency in sorted(histogram.items()):
            cumulative += frequency
            for label, fraction in (("p50", 0.5), ("p95", 0.95)):
                if label not in quantiles and cumulative >= math.ceil(count * fraction):
                    quantiles[label] = seconds
        metrics[name] = dict(mean=totals[name] / count if count else None, **quantiles)
    metrics["routeLength"] = dict(mean=totals["routeLength"] / count if count else None)
    return dict(
        states=dict(states),
        vehicle_types=dict(classes),
        completed_metrics=metrics,
        quantile_resolution_seconds=1,
    )


def simulation_statistics(path):
    if not path or not Path(path).is_file():
        return dict(available=False)
    result = dict(available=True)
    for tag in ("vehicles", "teleports", "safety", "vehicleTripStatistics"):
        for element in elements(path, (tag,)):
            result[tag] = {key: float(value) for key, value in element.attrib.items()}
    return result


def summary_statistics(path):
    """Summary counters have different semantics: never sum cumulative counts."""
    rows = []
    peak = Counter()
    last = {}
    for step in elements(path, ("step",)):
        row = {
            key: number(step, key) for key in ("time", "running", "waiting", "halting")
        }
        rows.append(row)
        for key in ("running", "waiting", "halting"):
            peak[key] = max(peak[key], row[key])
        last = dict(step.attrib)
    return dict(peak=dict(peak), final=last, series=rows)


def edge_statistics(path):
    stats = {}
    for edge in elements(path, ("edge",)):
        samples = number(edge, "sampledSeconds")
        if samples <= 0 or edge.get("id", "").startswith(":"):
            continue
        identifier = edge.get("id")
        row = stats.setdefault(
            identifier,
            dict(
                id=identifier,
                sampledSeconds=0.0,
                speed_sum=0.0,
                timeLoss=0.0,
                waitingTime=0.0,
                entered=0.0,
                left=0.0,
                teleported=0.0,
                worst_interval_timeLoss=0.0,
                worst_interval_begin=None,
                worst_interval_end=None,
            ),
        )
        row["sampledSeconds"] += samples
        row["speed_sum"] += samples * number(edge, "speed")
        for name in ("timeLoss", "waitingTime", "entered", "left", "teleported"):
            row[name] += number(edge, name)
        if number(edge, "timeLoss") > row["worst_interval_timeLoss"]:
            interval = edge.getparent()
            row.update(
                worst_interval_timeLoss=number(edge, "timeLoss"),
                worst_interval_begin=number(interval, "begin"),
                worst_interval_end=number(interval, "end"),
            )
    for row in stats.values():
        row["mean_speed_m_s"] = row.pop("speed_sum") / row["sampledSeconds"]
    return sorted(stats.values(), key=lambda row: (-row["timeLoss"], row["id"]))


def network_shapes(path, selected):
    shapes = {}
    for edge in elements(path, ("edge",)):
        if edge.get("id") not in selected:
            continue
        lane = edge.find("lane")
        text = edge.get("shape") or (lane.get("shape") if lane is not None else None)
        if text:
            shapes[edge.get("id")] = [
                tuple(map(float, point.split(",")[:2])) for point in text.split()
            ]
    return shapes


@serialized_plot
def quality_figure(report, edges, network, path):
    fig = Figure(figsize=(14, 8), facecolor="#f6f8fc", layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    states = report["trips"]["states"]
    axes[0, 0].bar(
        list(states),
        list(states.values()),
        color=["#137c8b", "#edae49", "#b54747", "#777"][: len(states)],
    )
    axes[0, 0].set(title="Recorded vehicle outcomes", ylabel="Vehicles")
    summary = report["summary"]["series"]
    for name, color in (
        ("running", "#137c8b"),
        ("waiting", "#d77731"),
        ("halting", "#9457a6"),
    ):
        axes[0, 1].plot(
            [row["time"] / 3600 for row in summary],
            [row[name] for row in summary],
            label=name,
            color=color,
        )
    axes[0, 1].set(
        title="Traffic and insertion queues",
        xlabel="Simulation time · hours",
        ylabel="Vehicles",
    )
    axes[0, 1].legend(frameon=False)
    worst = edges[:12][::-1]
    axes[1, 0].barh(
        [row["id"] for row in worst],
        [row["timeLoss"] / 3600 for row in worst],
        color="#137c8b",
    )
    axes[1, 0].set(
        title="Largest accumulated delays", xlabel="Total vehicle-hours lost"
    )
    shapes = network_shapes(network, {row["id"] for row in edges})
    from matplotlib.collections import LineCollection

    by_id = {row["id"]: row for row in edges}
    values = [math.log1p(by_id[key]["timeLoss"]) for key in shapes]
    lines = LineCollection(
        list(shapes.values()), array=values, cmap="magma_r", linewidths=1
    )
    axes[1, 1].add_collection(lines)
    axes[1, 1].autoscale()
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set(title="Delay on used regional roads")
    axes[1, 1].set_axis_off()
    if values:
        fig.colorbar(
            lines, ax=axes[1, 1], label="log(1 + total seconds lost)", shrink=0.7
        )
    for axis in axes.flat:
        axis.set_facecolor("white")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
    fig.suptitle(
        "SUMO | Simulation quality", fontsize=21, weight="bold", x=0.02, ha="left"
    )
    fig.savefig(path, dpi=150)
    fig.clear()


def generate_quality(result, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    edges = edge_statistics(result["edge_data"])
    report = dict(
        model=result["model"],
        trips=trip_statistics(result["tripinfo"]),
        statistics=simulation_statistics(result.get("statistics")),
        summary=summary_statistics(result["summary"]),
        bottleneck_count=len(edges),
        bottlenecks=edges[:25],
        sources={
            key: result.get(key)
            for key in ("tripinfo", "statistics", "summary", "edge_data")
        },
        notes=[
            "Trip quantiles use nearest-second bins and completed trips only.",
            "Teleport totals come from statistic-output; missing totals are unavailable, not zero.",
            "Bottlenecks rank normal edges by total timeLoss over all vehicles and intervals.",
            "Mean edge speed is weighted by sampled vehicle-seconds.",
            "These outputs identify symptoms, not the cause of a junction bottleneck.",
        ],
    )
    fields = list(edges[0]) if edges else ["id", "timeLoss"]
    with (destination / "bottlenecks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(edges)
    (destination / "quality.json").write_text(json.dumps(report, indent=2) + "\n")
    quality_figure(report, edges, result["network"], destination / "quality.png")
    teleports = report["statistics"].get("teleports", {}).get("total", "unavailable")
    states = report["trips"]["states"]
    state_text = " · ".join(
        f"{escape(key)}: {value:,}" for key, value in states.items()
    )
    rows = "".join(
        f"<tr><td>{escape(row['id'])}</td><td>{row['timeLoss'] / 3600:.2f}</td>"
        f"<td>{row['mean_speed_m_s'] * 3.6:.1f}</td><td>{row['teleported']:g}</td></tr>"
        for row in edges[:25]
    )
    metric_rows = []
    labels = {
        "duration": "Travel time (s)",
        "waitingTime": "Stopped time (s)",
        "timeLoss": "Time lost (s)",
        "departDelay": "Departure delay (s)",
        "routeLength": "Route length (m)",
    }
    for name, values in report["trips"]["completed_metrics"].items():
        cells = "".join(
            f"<td>{values[key]:.1f}</td>"
            if values.get(key) is not None
            else "<td>—</td>"
            for key in ("mean", "p50", "p95")
        )
        metric_rows.append(f"<tr><td>{labels[name]}</td>{cells}</tr>")
    metrics = "".join(metric_rows)
    events = " · ".join(
        f"{escape(group)} {escape(key)}: {value:g}"
        for group in ("teleports", "safety")
        for key, value in report["statistics"].get(group, {}).items()
    )
    notes = "".join(f"<li>{escape(note)}</li>" for note in report["notes"])
    (
        destination / "index.html"
    ).write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>SUMO simulation quality</title><style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;color:#203047;padding:0 20px;background:#f6f8fc}} img{{width:100%}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}a{{color:#137c8b}}</style>
<h1>SUMO simulation quality</h1><p>{escape(result["model"])} · {state_text}</p>
<p>Total teleports: <strong>{teleports}</strong>. See <a href="quality.json">all statistics</a> and <a href="bottlenecks.csv">all bottlenecks</a>.</p>
<p>{events}</p><h2>Completed trips</h2><table><tr><th>Measure</th><th>Mean</th><th>Median</th><th>95th percentile</th></tr>{metrics}</table>
<img src="quality.png" alt="Vehicle outcomes, traffic queues and road delay map">
<h2>Largest accumulated delays</h2><table><tr><th>Edge</th><th>Vehicle-hours lost</th><th>Mean speed (km/h)</th><th>Edge teleports</th></tr>{rows}</table><ul>{notes}</ul></html>""")
    return dict(
        directory=str(destination),
        html=str(destination / "index.html"),
        json=str(destination / "quality.json"),
        figure=str(destination / "quality.png"),
    )
