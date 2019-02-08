import numpy as np

from scipy.integrate import RK23
from IPython.display import clear_output

from . import siay
from .data import ChunkedDataSaver

class Integrator:
    def __init__(
            self, model, init_conditions, data_handler = None,
            init_step_size = siay / 10.0):

        if data_handler is None:
            data_handler = ChunkedDataSaver()

        self.model = model
        self.derivs = model.make_derivs()
        self.init_conditions = init_conditions
        self.h_t = []
        self.h_y = []

        self.data_handler = data_handler
        self.data_handler.initialized(self)

        self.setup_rk23(init_step_size)

    @staticmethod
    def restart(data):
        data.model.restart(data.ts[-1], data.ys[-1])
        out = Integrator(
            data.model,
            (data.ts[-1], data.ys[-1]),
            ChunkedDataSaver(existing_folder = data.folder_name)
        )
        out.h_t = data.ts.tolist()
        out.h_y = [data.ys[i] for i in range(data.ys.shape[0])]
        return out

    def setup_rk23(self, init_step_size):
        init_t, init_y = self.init_conditions
        self.rk23 = RK23(
            self.derivs,
            init_t,
            init_y,
            1e50,
            atol = self.model.cfg['timestep_tol'],
            rtol = self.model.cfg['timestep_tol']
        )
        self.rk23.h_abs = init_step_size

    def step_idx(self):
        return len(self.h_t)

    def integrate(self, n_steps, until = None, display_fnc = None, display_interval = 1):
        if display_fnc is None:
            display_fnc = lambda: print(self.h_t[-1])

        # import time
        for i in range(n_steps):
            # start = time.time()
            if until is not None and integrator.t > until:
                return
            assert(self.rk23.step() == None)
            new_t = self.rk23.t
            new_y = self.rk23.y.copy()
            self.h_t.append(new_t)
            self.h_y.append(new_y)
            self.model.post_step(self.h_t, self.h_y, self.rk23)

            if i % display_interval == 0:
                display_fnc(self)
            self.data_handler.stepped(self)
            # for i in range(20):
            #     print('full step', time.time() - start)
