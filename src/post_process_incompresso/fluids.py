"""Velocity utilities for 2-D incompreso ``.npz`` snapshots."""

import numpy as np

def mesh(snapshot):
    with np.load(snapshot) as data0:
        xu = data0["xu"]
        yu = data0["yu"]
        xv = data0["xv"]
        yv = data0["yv"]
        xp = data0["xp"]
        yp = data0["yp"]
    #print(xu, xv, xp)

    # Create 2D meshgrid for plotting - use pressure grid as common
    Xp, Yp = np.meshgrid(xp[1:-1], yp[1:-1])
    Xu, Yu = np.meshgrid(xu[1:-1], yu[1:-1])
    Xv, Yv = np.meshgrid(xv[1:-1], yv[1:-1])
    print(f"Meshgrid shapes: Xp={Xp.shape}, Xu={Xu.shape}, Xv={Xv.shape}")

    nx_u = len(xu) - 2
    ny_u = len(yu) - 2
    nx_v = len(xv) - 2
    ny_v = len(yv) - 2

    print(f"Grid sizes: u({nx_u} x {ny_u}), v({nx_v} x {ny_v})")
    return Xu, Yu, Xv, Yv, Xp, Yp, nx_u, ny_u, nx_v, ny_v

def max_velocity_magnitude(snapshots):
    if not snapshots:
        raise ValueError("snapshots must contain at least one file")

    u_max = 0
    v_max = 0
    _, _, _, _, _, _, nx_u, ny_u, nx_v, ny_v = mesh(snapshots[0])
    n_u = nx_u * ny_u
    n_v = nx_v * ny_v

    for s in snapshots:
        with np.load(s) as data:
            q = np.asarray(data["q"]).ravel()
        if q.size < n_u + n_v:
            raise ValueError(
                f"{s} contains {q.size} velocity values; "
                f"expected at least {n_u + n_v}"
            )

        u = q[:n_u].reshape(ny_u, nx_u)
        v = q[n_u:n_u + n_v].reshape(ny_v, nx_v)
        u_abs = np.max(np.abs(u))
        v_abs = np.max(np.abs(v))
        if u_abs > u_max:
            u_max = u_abs
        if v_abs > v_max:
            v_max = v_abs
    return u_max, v_max


def max_pressure(snapshots):
    if not snapshots:
        raise ValueError("snapshots must contain at least one file")

    p_max = -np.inf
    for s in snapshots:
        with np.load(s) as data:
            p = data["p"]
        max_p = np.max(p)
        if max_p > p_max:
            p_max = max_p
    return p_max


def vorticity(
    snapshots,
    config_path,
    output_dir=None,
    xlim=(-3.0, 7.0),
    ylim=(-3.0, 3.0),
    vmax=10.0,
    dataset_name=None,
):
    """Compute and save vorticity plots for a sequence of simulation snapshots."""
    import sys
    from pathlib import Path
    import re

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    incompreso_source = Path.home() / "incompreso-fork" / "src"
    if incompreso_source.is_dir() and str(incompreso_source) not in sys.path:
        sys.path.insert(0, str(incompreso_source))

    from incompreso import parse_input_file
    from incompreso.backend import to_numpy, xp
    from incompreso.utils.post_process import compute_vorticity

    snapshots = [Path(snapshot) for snapshot in snapshots]
    if not snapshots:
        raise ValueError("snapshots must contain at least one file")
    if vmax <= 0:
        raise ValueError("vmax must be greater than zero")
    if xlim is not None and xlim[0] >= xlim[1]:
        raise ValueError("xlim must be an increasing (minimum, maximum) pair")
    if ylim is not None and ylim[0] >= ylim[1]:
        raise ValueError("ylim must be an increasing (minimum, maximum) pair")

    simulation = parse_input_file(str(config_path))
    mesh = simulation["mesh"]
    bcs = simulation["bcs"]
    frame_dir = Path(output_dir) if output_dir is not None else snapshots[0].parent / "vorticity_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    written_frames = []
    for snapshot_path in snapshots:
        match = re.fullmatch(r"snapshot_(\d+)\.npz", snapshot_path.name)
        if match is None:
            raise ValueError(f"Not a numbered snapshot file: {snapshot_path}")

        with np.load(snapshot_path) as snapshot:
            time = float(snapshot["t"].item()) if "t" in snapshot.files else 0.0
            q = xp.asarray(snapshot["q"])
            omega, x_grid, y_grid = compute_vorticity(time, q, mesh, bcs)
            xi = snapshot["xi"].copy() if "xi" in snapshot.files else None
            eta = snapshot["eta"].copy() if "eta" in snapshot.files else None

        omega = to_numpy(omega)
        x_grid = to_numpy(x_grid)
        y_grid = to_numpy(y_grid)

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        try:
            image = ax.pcolormesh(
                x_grid,
                y_grid,
                omega,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                shading="auto",
            )
            if xi is not None and eta is not None:
                center_x, center_y = np.mean(xi), np.mean(eta)
                order = np.argsort(
                    np.arctan2(eta - center_y, xi - center_x)
                )
                outline_x = xi[order]
                outline_y = eta[order]
                ax.plot(
                    np.append(outline_x, outline_x[0]),
                    np.append(outline_y, outline_y[0]),
                    color="black",
                    lw=0.8,
                )
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_aspect("equal")
            title = (
                f"Vorticity: {dataset_name} | t = {time:.3f}"
                if dataset_name
                else f"t = {time:.3f}"
            )
            ax.set_title(title)
            ax.set_xlabel("x / D")
            ax.set_ylabel("y / D")
            fig.colorbar(image, ax=ax, shrink=0.9, label=r"$\omega_z\, D/U_\infty$")

            name_suffix = f"_{dataset_name}" if dataset_name else ""
            output_path = frame_dir / (
                f"vorticity{name_suffix}_{int(match.group(1)):06d}.png"
            )
            fig.savefig(output_path, dpi=150)
        finally:
            plt.close(fig)

        written_frames.append(output_path)
        print(f"Wrote {output_path}")

    return written_frames