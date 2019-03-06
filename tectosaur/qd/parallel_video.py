import os
import psutil
import multiprocessing
import matplotlib.cbook
import matplotlib.pyplot as plt
import numpy as np

import tectosaur.qd as qd
from tectosaur.qd.plot_config import meade03_socket_idx
from tectosaur.qd.data import skip_existing_prefixed_folders

class DataChunk:
    def __init__(self, datadir, chunk_num):
        initial_data = np.load(os.path.join(datadir, 'initial_data.npy'))
        self.m, self.cfg, self.init_conditions = initial_data
        self.model = qd.TopoModel(self.m, self.cfg)

        filename = os.path.join(datadir, str(chunk_num) + '.npy')
        if os.path.exists(filename):
            chunk = np.load(filename)
            n_steps = len(chunk[0])
            self.ts = np.array(chunk[0])
            self.ys = np.array(chunk[1])
        else:
            self.ts = None
            self.ys = None

def init(queue):
    global idx
    idx = queue.get()

def process(plotting_data):
    plot_fnc, datadir, start_step, steps_paths = plotting_data

    global idx
    psutil.Process().cpu_affinity([idx])

    which = start_step + (idx + 1) * 100
    data = DataChunk(datadir, which)
    if data.ts is None:
        return

    qdp  = qd.plotting.QDPlotData(data)

    for step_idx, filepath in steps_paths:
        local_step_idx = step_idx - which + 100
        if not (0 <= local_step_idx < 100):
            continue
        # print(step_idx, filepath)
        success = False
        retries = 1000
        while not success:
            try:
                plot_fnc(qdp, local_step_idx, filepath)
                success = True
            except matplotlib.cbook.Locked.TimeoutError:
                print("Latex lock, retry.")
                plt.close()
                retries -= 1
                if retries == 0:
                    raise


def get_frame_name(frame_idx, n_frames):
    digits = len(str(n_frames))
    return '%0*d' % (digits, frame_idx)

def video_test(plot_fnc, datadir, step_idx):
    which = 100 * (int(np.ceil(step_idx // 100)) + 1)
    local_step_idx = step_idx - which + 100
    data = DataChunk(datadir, which)
    qdp  = qd.plotting.QDPlotData(data)
    plot_fnc(qdp, local_step_idx, None)

def parallel_video(plot_fnc, datadir, n_cores, steps_to_plot, video_prefix):
    if video_prefix is not None:
        video_name = skip_existing_prefixed_folders(video_prefix)
        os.makedirs(video_name)

    n_frames = len(steps_to_plot)
    steps_paths= []
    for frame_idx in range(n_frames):
        step = steps_to_plot[frame_idx]
        frame_name = get_frame_name(frame_idx, n_frames)
        filepath = None if video_prefix is None else f'{video_name}/{frame_name}.png'
        steps_paths.append((step, filepath))

    ids = list(range(n_cores))
    manager = multiprocessing.Manager()
    idQueue = manager.Queue()
    for i in ids:
        idQueue.put(i)

    p = multiprocessing.Pool(n_cores, init, (idQueue,))

    loops = int(np.ceil(np.max(steps_to_plot) / (n_cores * 100)))
    # print('loops', loops)
    for i in range(loops):
        start_step = n_cores * 100 * i
        p.map(process, [(plot_fnc, datadir, start_step, steps_paths)] * n_cores)

    return video_name
