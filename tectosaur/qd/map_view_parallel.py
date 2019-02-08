import numpy as np
import matplotlib
import subprocess
import os
import uuid
import cloudpickle
import multiprocessing
from shapely.geometry import Polygon
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata
from matplotlib.collections import LineCollection
from matplotlib import cm
from boundary import get_boundary_loop

LINE_WIDTH = 0.5
BUFFER_KM = 50
BUFFER_COLOR = "lightgray"
# TIME_IDX = 800
VMIN = -11
VMAX = -1
WHISKER_SCALE = 1e1
SIAY = 60 * 60 * 24 * 365
USE_N_CORES = 30

# Read Ben's data for Cascadia
pts, tris, t, slip_all, state_all = np.load("data_for_brendan.npy")

# Rescale to km with a LLC origin
pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / 1e3
pts[:, 1] = (pts[:, 1] - pts[:, 1].min()) / 1e3
pts[:, 2] = (pts[:, 2]) / 1e3

dt_vec = np.diff(t)
slip_magnitude_all = np.sqrt(
    slip_all[:, :, 0] ** 2 + slip_all[:, :, 1] ** 2 + slip_all[:, :, 2] ** 2
)
slip_diff_all = np.diff(slip_magnitude_all, axis=0)
slip_rate_all = slip_diff_all / dt_vec[:, np.newaxis]
slip_rate_log = np.log10(np.abs(slip_rate_all))
slip_rate_x = np.diff(slip_all[:, :, 0], axis=0) / dt_vec[:, np.newaxis]
slip_rate_y = np.diff(slip_all[:, :, 1], axis=0) / dt_vec[:, np.newaxis]


def plotter_executor(data):
    serialized_fnc, frame_idx, filepath = data
    print("frame =", frame_idx)
    fnc = cloudpickle.loads(serialized_fnc)
    fnc(frame_idx, filepath)


def make_video(video_name, n_frames, plotter_fnc, framerate=60):
    os.makedirs(video_name)
    digits = len(str(n_frames))
    pool = multiprocessing.Pool(processes=USE_N_CORES)
    serialized_fnc = cloudpickle.dumps(plotter_fnc)
    job_data = []
    for frame_idx in range(n_frames):
        frame_name = "%0*d" % (digits, frame_idx)
        filepath = f"{video_name}/{frame_name}.png"
        job_data.append((serialized_fnc, frame_idx, filepath))
    pool.map(plotter_executor, job_data)

    cmd = [
        "ffmpeg",
        "-framerate",
        str(framerate),
        "-i",
        f"{video_name}/%0{digits}d.png",
        "-c:v",
        "libx264",
        "-r",
        "30",
        "-y",
        "-v",
        "32",
        video_name + ".png",
    ]
    print("running", '"' + " ".join(cmd) + '"')
    for line in execute(cmd):
        print(line, end="")


def execute(cmd):
    popen = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stderr.readline, ""):
        yield stdout_line
    popen.stderr.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def format_title(t_orig):
    t_current = t_orig.copy()
    year = np.floor(t_current / SIAY).astype(int)

    t_current = t_current - year * SIAY
    day = np.floor(t_current / (60 * 60 * 24)).astype(int)

    t_current = t_current - day * (60 * 60 * 24)
    hour = np.floor(t_current / (60 * 60)).astype(int)

    t_current = t_current - hour * (60 * 60)
    minute = np.floor(t_current / (60)).astype(int)

    t_current = t_current - minute * 60
    second = np.floor(t_current).astype(int)

    title_string = (
        "y:"
        + str(year).zfill(4)
        + " d:"
        + str(day).zfill(3)
        + " h:"
        + str(hour).zfill(2)
        + " m:"
        + str(minute).zfill(2)
        + " s:"
        + str(second).zfill(2)
    )

    return title_string


def get_boundary(tris):
    loop = get_boundary_loop(tris)[0]
    boundary_x = np.zeros(len(loop))
    boundary_y = np.zeros(len(loop))
    for i in range(len(loop)):
        P = pts[loop[i]]
        boundary_x[i] = P[0]
        boundary_y[i] = P[1]
    return boundary_x, boundary_y


def plot_boundary(tris, outline):
    plt.fill(
        [
            np.min(pts[:, 0]) - BUFFER_KM,
            np.max(pts[:, 0]) + BUFFER_KM,
            np.max(pts[:, 0]) + BUFFER_KM,
            np.min(pts[:, 0]) - BUFFER_KM,
        ],
        [
            np.min(pts[:, 1]) - BUFFER_KM,
            np.min(pts[:, 1]) - BUFFER_KM,
            np.max(pts[:, 1]) + BUFFER_KM,
            np.max(pts[:, 1]) + BUFFER_KM,
        ],
        BUFFER_COLOR,
        zorder=-1,
    )

    if outline:
        plt.plot(
            np.append(boundary_x, boundary_x[0]),
            np.append(boundary_y, boundary_y[0]),
            "-k",
            zorder=100,
            linewidth=0.5,
        )


def plot_whiskers(idx):
    """ Plot small whiskers indicated slip orientation """
    nx = 5
    ny = 15
    x_vec = np.linspace(np.min(pts[:, 0]), np.max(pts[:, 0]), nx)
    y_vec = np.linspace(np.min(pts[:, 1]), np.max(pts[:, 1]), ny)
    x_grid, y_grid = np.meshgrid(x_vec, y_vec)
    u_x = griddata(
        (pts[:, 0], pts[:, 1]), slip_rate_x[idx, :], (x_grid, y_grid), method="cubic"
    )
    u_y = griddata(
        (pts[:, 0], pts[:, 1]), slip_rate_y[idx, :], (x_grid, y_grid), method="cubic"
    )
    u_y[np.isnan(u_y)] = 0
    u_x[np.isnan(u_x)] = 0

    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    u_x = u_x.flatten()
    u_y = u_y.flatten()

    # Select only those inside fault boundary
    boundary_x, boundary_y = get_boundary(tris)
    boundary_points = list(zip(boundary_x, boundary_y))
    polygon = Polygon(boundary_points)

    for i in range(0, u_x.size):
        if polygon.contains(Point((x_grid[i], y_grid[i]))):

            vel_mag = np.sqrt(u_x[i] ** 2 + u_y[i] ** 2)
            if vel_mag > 1e-14:
                ux = u_x[i] / vel_mag
                uy = u_y[i] / vel_mag

                plt.plot(
                    [x_grid[i], x_grid[i] + WHISKER_SCALE * ux],
                    [y_grid[i], y_grid[i] + WHISKER_SCALE * uy],
                    "-k",
                    zorder=1000,
                    linewidth=LINE_WIDTH,
                )
                plt.plot(
                    [x_grid[i], x_grid[i] - WHISKER_SCALE * ux],
                    [y_grid[i], y_grid[i] - WHISKER_SCALE * uy],
                    "-k",
                    zorder=1000,
                    linewidth=LINE_WIDTH,
                )


def f(idx, filepath):
    """ Main rendering function """
    plt.figure(figsize=(6, 6))
    contour_levels = np.linspace(VMIN, VMAX, 11)
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(1, 1, 1, aspect=1)
    mesh = plt.tricontourf(
        pts[:, 0],
        pts[:, 1],
        tris,
        slip_rate_log[idx, :],
        contour_levels,
        cmap="plasma",
        vmin=VMIN,
        vmax=VMAX,
        extend="both",
    )

    plot_boundary(tris, outline=False)
    plot_whiskers(idx)
    plt.gca().set_aspect("equal")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.xticks([0, 400])
    plt.yticks([0, 400, 800, 1200])
    plt.xlim([np.min(pts[:, 0]) - BUFFER_KM, np.max(pts[:, 0]) + BUFFER_KM])
    plt.ylim([np.min(pts[:, 1]) - BUFFER_KM, np.max(pts[:, 1]) + BUFFER_KM])
    plt.title(format_title(t[idx]), fontsize=10)

    m = plt.cm.ScalarMappable(cmap="plasma")
    m.set_array(slip_rate_log)
    m.set_clim(VMIN, VMAX)
    cbar_axes = fig.add_axes([0.35, 0.1, 0.015, 0.3])
    cbar = plt.colorbar(m, boundaries=contour_levels, cax=cbar_axes)
    cbar.set_ticks(np.array([-11, -9, -7, -5, -3, -1]))
    cbar.set_label("log v")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)


def main():
    make_video("map_animation_" + uuid.uuid4().hex, 1999, f)


if __name__ == "__main__":
    main()
