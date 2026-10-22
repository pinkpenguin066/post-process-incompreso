import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def centroid_position_velocity(snapshots):
    times = []
    y_position = []
    y_velocity = []

    for snapshot in snapshots:
        root = ET.parse(snapshot).getroot()
        points_node = root.find('.//Points/DataArray')
        velocity_node = root.find('.//PointData/DataArray[@Name="velocity"]')
        time_node = root.find('.//FieldData/DataArray[@Name="TimeValue"]')
        if points_node is None or velocity_node is None or time_node is None:
            raise ValueError(
                f'Missing points, velocity, or TimeValue in {snapshot}'
            )
        if time_node.text is None:
            raise ValueError(f'Missing TimeValue content in {snapshot}')

        n_point_components = int(
            points_node.attrib.get('NumberOfComponents', '3')
        )
        n_velocity_components = int(
            velocity_node.attrib.get('NumberOfComponents', '3')
        )
        if n_point_components < 2 or n_velocity_components < 2:
            raise ValueError(f'Expected at least two components in {snapshot}')

        points = np.fromstring(
            points_node.text or '', sep=' '
        ).reshape(-1, n_point_components)
        body_velocity = np.fromstring(
            velocity_node.text or '', sep=' '
        ).reshape(-1, n_velocity_components)
        if points.shape[0] == 0 or body_velocity.shape[0] != points.shape[0]:
            raise ValueError(
                f'Points and velocity must contain the same nonzero number '
                f'of entries in {snapshot}'
            )
        times.append(float(time_node.text))
        y_position.append(np.mean(points[:, 1]))
        y_velocity.append(np.mean(body_velocity[:, 1]))

    if not times:
        raise ValueError('At least one snapshot is required')

    order = np.argsort(times, kind='stable')
    return (
        np.asarray(times)[order],
        np.asarray(y_position)[order],
        np.asarray(y_velocity)[order],
    )


def multi_position_velocity(runs):
    """Return time, position, and velocity arrays for named simulation runs."""
    if not runs:
        raise ValueError("At least one run is required")

    return {
        run_name: centroid_position_velocity(body_snapshots)
        for run_name, body_snapshots in runs.items()
    }


def square_extract(data_location, ax=None):
    input_path = Path(data_location) / "square.npz"
    show_plot = ax is None
    if ax is None:
        _, ax = plt.subplots()

    with np.load(input_path) as data:
        if "xi" not in data or "eta" not in data:
            raise ValueError(
                f"{input_path} must contain 'xi' and 'eta' arrays"
            )
        xi = np.asarray(data["xi"])
        eta = np.asarray(data["eta"])

    if xi.ndim != 1 or eta.ndim != 1 or xi.shape != eta.shape:
        raise ValueError("'xi' and 'eta' must be one-dimensional arrays of equal length")
    if xi.size == 0:
        raise ValueError(f"{input_path} contains no marker positions")

    print(f"Loaded {xi.size} marker positions from {input_path}")
    center_x, center_y = np.mean(xi), np.mean(eta)
    order = np.argsort(np.arctan2(eta - center_y, xi - center_x))
    outline_x = xi[order]
    outline_y = eta[order]
    ax.plot(
        np.append(outline_x, outline_x[0]),
        np.append(outline_y, outline_y[0]),
        color="C0",
    )
    ax.set_xlabel("xi (x-coordinate)")
    ax.set_ylabel("eta (y-coordinate)")
    ax.set_title(f"Immersed-body geometry ({xi.size} points)")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.grid(True)
    if show_plot:
        plt.show()
    return ax