"""Diagnostic renders of a label against the image it was produced from.

Numbers describe a label; they do not show it. The failures that matter most in
the un-annotated majority of a cohort - a label that leaked into a neighbour,
lost a lobe, or sits a few voxels off the edge it should follow - are obvious on
sight and awkward to express as a statistic. A rendered panel turns them into
something reviewable, by a person or by a model that can look at an image.

Three choices here are deliberate.

Labels are drawn as outlines, not fills. A filled overlay hides the intensities
underneath, which are exactly what the reviewer needs in order to judge whether
the boundary follows a real edge.

Slices are spread across the label's own extent rather than the volume's. A
structure occupying a third of the field would otherwise be reviewed mostly
through empty tissue, and a lost lobe at one end would never appear.

Several labels can share a panel. Drawing a propagated label and its correction
together shows what the correction actually did, which is more informative than
either alone and is what makes a before-and-after judgement possible.

Rendering is deliberately the end of this module's responsibility. It writes an
image and reports what is in it; nothing here interprets the result. Judging the
render is the reviewer's job, and keeping that outside the package is what
allows the package to stay deterministic and free of a model dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..io.volumes import Volume

#: Outline colours, in the order labels are supplied. Chosen to stay
#: distinguishable from one another and from greyscale tissue.
OUTLINE_COLOURS = ("#4CC9F0", "#F72585", "#FFD60A", "#80ED99")

#: Intensity percentiles used to window the greyscale display. The upper bound
#: is below 100 so a few bright voxels cannot flatten the whole image.
WINDOW = (1.0, 99.5)

#: Fraction of the label's extent trimmed from each end before choosing slices,
#: so no slice lands on the single voxel where a structure begins.
EDGE_TRIM = 0.08

AXIS_NAMES = ("axis 0", "axis 1", "axis 2")


@dataclass(frozen=True)
class Panel:
    """A rendered review figure and a description of what it shows."""

    path: str
    specimen: str
    #: Label names drawn, in the order their colours were assigned.
    labels: list[str]
    #: Outline colour per label name, so a reviewer can be told which is which.
    colours: dict[str, str]
    #: Slice indices drawn, keyed by axis.
    slices: dict[int, list[int]]

    def caption(self) -> str:
        """A description of the figure, for handing to whoever reviews it."""
        drawn = ", ".join(f"{n} in {self.colours[n]}" for n in self.labels)
        rows = "; ".join(
            f"{AXIS_NAMES[a]} at {', '.join(str(i) for i in idx)}"
            for a, idx in sorted(self.slices.items())
        )
        return (
            f"{self.specimen}: greyscale specimen intensities with label "
            f"outlines ({drawn}). Rows are array axes, columns are slices "
            f"spread through the label's extent - {rows}. Array axes as stored; "
            f"no anatomical reorientation is applied."
        )


def _window(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    sample = finite[finite > 0]
    if sample.size == 0:
        sample = finite
    if sample.size == 0:
        return 0.0, 1.0
    lo, hi = (float(np.percentile(sample, p)) for p in WINDOW)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _slice_positions(mask: np.ndarray, axis: int, count: int) -> list[int]:
    """Indices spread through the part of an axis the label actually occupies."""
    present = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
    if present.size == 0:
        return []
    lo, hi = int(present[0]), int(present[-1])
    span = hi - lo
    trim = int(round(span * EDGE_TRIM))
    lo, hi = lo + trim, hi - trim
    if hi <= lo:
        return [int((lo + hi) // 2)]
    return [int(round(v)) for v in np.linspace(lo, hi, count)]


def render_panel(
    image: Volume,
    labels: Mapping[str, Volume] | Volume,
    path: str | Path,
    *,
    specimen: str = "",
    axes: Sequence[int] = (0, 1, 2),
    per_axis: int = 3,
    dpi: int = 110,
) -> Panel:
    """Render label outlines over specimen intensities and write a PNG.

    Args:
        image: Specimen intensities.
        labels: One label, or a mapping of name to label. Several labels are
            drawn together in different colours, which is how a correction is
            compared against what it was applied to.
        path: Destination PNG.
        specimen: Identifier, written into the figure and the caption.
        axes: Array axes to slice along, one row each.
        per_axis: Slices per axis, one column each.

    Returns:
        A :class:`Panel` naming the file, the colour assigned to each label, and
        the slice indices drawn, so a reviewer can be told exactly what they are
        looking at.

    Raises:
        ValueError: If no label contains any voxels, since there would be
            nothing to centre the review on.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(labels, Volume):
        labels = {"label": labels}
    if not labels:
        raise ValueError("at least one label is required")
    for name, volume in labels.items():
        image.assert_compatible(volume)

    masks = {name: volume.data > 0 for name, volume in labels.items()}
    union = np.zeros(image.data.shape, dtype=bool)
    for m in masks.values():
        union |= m
    if not union.any():
        raise ValueError("every supplied label is empty; nothing to review")

    colours = {
        name: OUTLINE_COLOURS[i % len(OUTLINE_COLOURS)]
        for i, name in enumerate(labels)
    }
    positions = {axis: _slice_positions(union, axis, per_axis) for axis in axes}
    positions = {a: p for a, p in positions.items() if p}

    rows = len(positions)
    cols = max(len(p) for p in positions.values())
    lo, hi = _window(image.data)

    fig, grid = plt.subplots(
        rows, cols, figsize=(3.1 * cols, 3.1 * rows), squeeze=False,
        facecolor="#0B0B0D",
    )

    for r, (axis, indices) in enumerate(sorted(positions.items())):
        for c in range(cols):
            ax = grid[r][c]
            ax.set_facecolor("#0B0B0D")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#2A2A30")
            if c >= len(indices):
                ax.axis("off")
                continue
            index = indices[c]
            plane = np.take(image.data, index, axis=axis).astype(float)
            ax.imshow(plane.T, cmap="gray", vmin=lo, vmax=hi,
                      origin="lower", interpolation="nearest", aspect="equal")
            for name, mask in masks.items():
                slab = np.take(mask, index, axis=axis)
                if slab.any():
                    ax.contour(slab.T.astype(float), levels=[0.5],
                               colors=[colours[name]], linewidths=0.9)
            ax.set_title(f"{AXIS_NAMES[axis]} · {index}", color="#9AA0A6",
                         fontsize=8, pad=4)

    handles = [
        plt.Line2D([], [], color=colours[n], linewidth=1.6, label=n) for n in labels
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, labelcolor="#D6D8DB", fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    title = specimen or "label review"
    fig.suptitle(title, color="#E8EAED", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    return Panel(
        path=str(destination),
        specimen=specimen,
        labels=list(labels),
        colours=colours,
        slices={a: list(p) for a, p in sorted(positions.items())},
    )
