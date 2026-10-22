import glob
import importlib
import importlib.util
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


from . import body as bd
from . import fluids as flu

from .post import post

class multi_post:
    def __init__(self, data_location, output_base_location):
        self.data_location = data_location
        self.output_base_location = output_base_location
        self.output_location = os.path.join(
            self.output_base_location, "result_comparison"
        )
    
    def muti_plot_body_position_velocity(
        self,
        lower_limit=None,
        upper_limit=None,
    ):
        if data_locations is None:
            data_locations = [self.data_location]
        elif isinstance(data_locations, (str, os.PathLike)):
            data_locations = [data_locations]
        output_location = output_location or self.output_location

        run_data = {}
        run_max_times = {}
        for data_location in data_locations:
            data_location = os.fspath(data_location)
            run_name = os.path.basename(os.path.normpath(data_location))
            if run_name in run_data:
                raise ValueError(f"Duplicate run name: {run_name}")
            if not os.path.isdir(data_location):
                raise FileNotFoundError(
                    f"Run data folder does not exist: {data_location}"
                )

            simulation_snapshots = glob.glob(
                os.path.join(data_location, "snapshot_*.npz")
            )
            simulation_snapshots = [
                path
                for path in simulation_snapshots
                if re.search(r"snapshot_(\d+)\.npz$", path)
            ]
            if not simulation_snapshots:
                raise ValueError(
                    f"No snapshot_*.npz files found in {data_location}"
                )

            snapshot_times = []
            for snapshot_path in simulation_snapshots:
                with np.load(snapshot_path) as snapshot:
                    if "t" not in snapshot.files:
                        raise ValueError(
                            f"Snapshot has no simulation time 't': "
                            f"{snapshot_path}"
                        )
                    time_value = np.asarray(snapshot["t"])
                    if time_value.size != 1:
                        raise ValueError(
                            f"Expected a scalar simulation time in "
                            f"{snapshot_path}"
                        )
                    snapshot_times.append(float(time_value.item()))
            run_max_times[run_name] = max(snapshot_times)

            run_post = Post(
                data_location,
                output_base_location=self.output_base_location,
            )
            body_snapshots = run_post.get_body_snapshot()
            body_snapshots = [
                path
                for path in body_snapshots
                if (
                    lower_limit is None
                    or lower_limit
                    <= int(re.search(r"snapshot_(\d+)_body\.vtp$", path).group(1))
                )
                and (
                    upper_limit is None
                    or int(re.search(r"snapshot_(\d+)_body\.vtp$", path).group(1))
                    <= upper_limit
                )
            ]
            if not body_snapshots:
                raise ValueError(
                    f"No body snapshots found in {data_location} "
                    "within the selected limits"
                )
            run_data[run_name] = bd.centroid_position_velocity(body_snapshots)

        common_max_time = min(run_max_times.values())
        print(f"Limiting comparison to t <= {common_max_time:g}")

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for run_name, (times, y_position, y_velocity) in run_data.items():
            times = np.asarray(times)
            y_position = np.asarray(y_position)
            y_velocity = np.asarray(y_velocity)
            within_common_time = times <= common_max_time
            axes[0].plot(
                times[within_common_time],
                y_position[within_common_time],
                "-",
                linewidth=1,
                label=run_name,
            )
            axes[1].plot(
                times[within_common_time],
                y_velocity[within_common_time],
                "-",
                linewidth=1,
                label=run_name,
            )

        axes[0].set_ylabel("non-dim y position")
        axes[0].set_title("Body position vs Time")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()
        axes[1].set_xlabel("Time unit")
        axes[1].set_ylabel("non-dim y velocity")
        axes[1].set_title("Body velocity vs Time")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()
        fig.tight_layout()

        os.makedirs(output_location, exist_ok=True)
        output_path = os.path.join(
            output_location, "body_position_velocity_comparison.png"
        )
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Comparison plot saved to {output_path}")
        return run_data