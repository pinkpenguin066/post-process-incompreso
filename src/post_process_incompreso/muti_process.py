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

from .post import Post

class muti_process:
    def __init__(self, data_location, output_location, upper_limit=None, lower_limit=None):
        self.data_location = data_location
        self.output_location = output_location
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
    
    def muti_plot_body_position_velocity(self):
        muti_data_location = self.data_location
        if isinstance(muti_data_location, (str, os.PathLike)):
            muti_data_location = [muti_data_location]
        output_location = self.output_location

        run_data = {}
        run_max_times = {}
        for data_location in muti_data_location:
            data_location = os.fspath(data_location)
            run_name = os.path.basename(os.path.normpath(data_location))
            if run_name in run_data:
                raise ValueError(f"Duplicate run name: {run_name}")
            if not os.path.isdir(data_location):
                raise FileNotFoundError(
                    f"Run data folder does not exist: {data_location}"
                )
            post = Post(
                data_location,
                output_base_location=output_location,
            )
            run_data[run_name] = post.body_position_velocity()
            run_max_times[run_name] = max(run_data[run_name][0]) 

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

        os.makedirs(self.output_location, exist_ok=True)
        output_path = os.path.join(
            self.output_location, "body_position_velocity_comparison.png"
        )
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Comparison plot saved to {output_path}")
        return run_data
                
