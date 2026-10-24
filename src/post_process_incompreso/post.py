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

class Post:
    def __init__(
        self,
        data_location,
        *,
        output_base_location=None,
        config_location=None,
        config_name=None,
        params_location=None,
        upper_limit=None,
        lower_limit=None,
    ):
        self.data_location = os.fspath(data_location)
        self.data_name = os.path.basename(os.path.normpath(self.data_location))
        self.output_base_location = os.fspath(
            output_base_location or os.path.dirname(self.data_location)
        )
        self.output_location = os.path.join(
            self.output_base_location, f"result_{self.data_name}"
        )
        self.vorticity_location = os.path.join(self.data_location, "vorticity")
        self.config_location = config_location
        self.config_name = config_name
        self.params_location = params_location
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit

    def __repr__(self) -> str:
        return f"Post(data_dir='{self.data_dir}')"
        
    def get_fluid_snapshot(self):
        snapshots = glob.glob(
            os.path.join(self.data_location, "snapshot_*.npz")
        )
        snapshots = [
            path
            for path in snapshots
            if re.search(r"snapshot_(\d+)\.npz$", path)
        ]
        snapshots.sort(
            key=lambda path: int(
                re.search(r"snapshot_(\d+)\.npz$", path).group(1)
            )
        )
        print("Files selected:", len(snapshots))
        snapshots = self.snapshot_limit(snapshots)
        return snapshots

    def get_body_snapshot(self):
        snapshots = glob.glob(
            os.path.join(self.data_location, "snapshot_*_body.vtp")
        )
        snapshots = [
            path
            for path in snapshots
            if re.search(r"snapshot_(\d+)_body\.vtp$", path)
        ]
        snapshots.sort(
            key=lambda path: int(
                re.search(r"snapshot_(\d+)_body\.vtp$", path).group(1)
            )
        )
        print("Body files selected:", len(snapshots))
        snapshots = self.snapshot_limit(snapshots)
        return snapshots

    def snapshot_limit(self, snapshots):
        snapshots = [
            path
            for path in snapshots
            if (
                self.lower_limit is None
                or self.lower_limit
                <= int(re.search(r"snapshot_(\d+)_body\.vtp$", path).group(1))
            )
            and (
                self.upper_limit is None
                or int(re.search(r"snapshot_(\d+)_body\.vtp$", path).group(1))
                <= self.upper_limit
            )
        ]
        if not snapshots:
            raise ValueError(
                f"No body snapshots found in {self.data_location} "
                "within the selected limits"
            )
        
        return snapshots
    
    def vorticity_calculation(
        self, config_location=None, config_name=None, output_dir=None
    ):
        config_location = config_location or self.config_location
        config_name = config_name or self.config_name
        if config_location is None or config_name is None:
            raise ValueError(
                "Set config_location and config_name when creating Post "
                "or calling vorticity_calculation."
            )

        flu.vorticity(
            self. get_fluid_snapshot(),
            config_path=os.path.join(config_location, config_name),
            output_dir=output_dir or self.vorticity_location,
            dataset_name=self.data_name,
        )

    def movie(self, image_path=None, output_file=None):
        import imageio.v3 as iio
        from PIL import Image

        image_path = image_path or os.path.join(
            self.vorticity_location, "vorticity_*.png"
        )
        output_file = output_file or os.path.join(
            self.output_location, "vorticity_movie.mp4"
        )
        file_names = sorted(glob.glob(image_path))

        if not file_names:
            print(f"No PNG files found matching {image_path!r}")
            return

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        images = []
        with Image.open(file_names[0]) as first_image:
            target_size = first_image.size

        for filename in file_names:
            with Image.open(filename) as image:
                if image.size != target_size:
                    print(f"Resizing {filename} to match other frames...")
                    image = image.resize(
                        target_size, Image.Resampling.LANCZOS
                    )
                images.append(np.array(image))

        iio.imwrite(output_file, images, extension=".mp4", fps=30)
        print(f"Success! Movie saved to {output_file}")

    def max_velocity_calculation(self):
        u, v = flu.max_velocity_magnitude(self. get_fluid_snapshot())
        print(f"Max velocity magnitude: u={u:.6e}, v={v:.6e}")
        return u, v

    def body_position_velocity(self):
        return bd.centroid_position_velocity(self.get_body_snapshot())

    def plot_body_position_velocity(self):
        t, y, v = self.body_position_velocity()
        os.makedirs(self.output_location, exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t, y, "-", color="tab:blue", linewidth=1)
        axes[0].set_ylabel("non-dim y position")
        axes[0].set_title(f"Body position vs Time - {self.data_name}")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[1].plot(t, v, "-", color="tab:blue", linewidth=1)
        axes[1].set_xlabel("Time unit")
        axes[1].set_ylabel("non-dim y velocity")
        axes[1].set_title(f"Body velocity vs Time - {self.data_name}")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                self.output_location,
                f"body_position_velocity_{self.data_name}.png",
            ),
            dpi=300,
        )
        plt.close(fig)

    def plot_square_geometry(self):
        os.makedirs(self.output_location, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 7))
        bd.square_extract(self.data_location, ax=ax)
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                self.output_location,
                f"square_geometry_{self.data_name}.png",
            ),
            dpi=300,
        )
        plt.close(fig)

    def param_txt(self):
        smd_path = os.path.join(self.data_location, "smd_history.npz")
        if not os.path.isfile(smd_path):
            raise FileNotFoundError(f"No smd_history.npz found in {self.data_location!r}")

        if self.params_location is None:
            params_module = importlib.import_module("compute_smd_params")
        else:
            params_path = Path(self.params_location)
            if params_path.is_dir():
                params_path = params_path / "compute_smd_params.py"
            if not params_path.is_file():
                raise FileNotFoundError(
                    f"compute_smd_params.py not found at {params_path}"
                )
            spec = importlib.util.spec_from_file_location(
                "post_compute_smd_params", params_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load parameters from {params_path}")
            params_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(params_module)

        params = params_module.PARAMS
        with np.load(smd_path, allow_pickle=False) as data:
            parameters = {
                key: data[key].item()
                for key in data.files
                if data[key].ndim == 0
            }

        rho_f = params["rho_f"]
        diameter = params["D"]
        parameters["rho_f"] = rho_f
        parameters["D"] = diameter
        reference_velocity = parameters["U_ref"]
        parameters["M_CG"] = (
            parameters["M_CG_nondim"] * rho_f * diameter**3
        )
        parameters["K_CG"] = (
            parameters["k_CG_nondim"]
            * rho_f
            * reference_velocity**2
            * diameter
        )
        parameters["C_CG"] = (
            parameters["c_CG_nondim"]
            * rho_f
            * reference_velocity
            * diameter**2
        )

        os.makedirs(self.output_location, exist_ok=True)
        param_file_path = os.path.join(
            self.output_location, f"param_{self.data_name}.txt"
        )
        with open(param_file_path, "w") as output_file:
            for name, value in parameters.items():
                output_file.write(f"{name}: {value}\n")
        print(f"Parameter file saved to {param_file_path}")
        return parameters

    def voltage(self, post_transit = 0):
        t, y , v = self. body_position_velocity()
        y_dim = y * 0.0192
        chi = 0.00023
        C_p = 4.3e-8
        tau = 1.1135
        voltage = chi / C_p / tau * y_dim
        after_t = t > post_transit
        if not np.any(after_t):
            raise ValueError(f'No voltage samples found after T = {post_transit}')
        voltage_rms_after_t = np.sqrt(np.mean(voltage[after_t] ** 2))

        os.makedirs(self.output_location, exist_ok=True)
        voltage_file_path = os.path.join(
            self.output_location, f"rms_voltage_{self.data_name}.txt"
        )
        with open(voltage_file_path, "w") as output_file:
            output_file.write(f"rms voltage after t = {post_transit}: {voltage_rms_after_t}\n")
        print(f"voltage file saved to {voltage_file_path}")

        plt.figure(figsize=(10, 4))
        plt.plot(t, voltage, '-', color='tab:green', linewidth=1.2)
        plt.xlabel('Time unit')
        plt.ylabel('Voltage [V]')
        plt.title(f'Voltage vs Time ({self.data_name})')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(self.output_location, dpi=300)
        plt.close()
        return voltage_rms_after_t