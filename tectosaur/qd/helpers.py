import numpy as np

from tectosaur.util.geometry import unscaled_normals

def tri_normal_info(m):
    unscaled_tri_normals = unscaled_normals(m.pts[m.tris])
    tri_size = np.linalg.norm(unscaled_tri_normals, axis = 1)
    tri_normals = unscaled_tri_normals / tri_size[:, np.newaxis]
    return unscaled_tri_normals, tri_size, tri_normals

from IPython.display import Audio, display, clear_output

rate = 16000.
def synth(f, duration):
    t = np.linspace(0., duration, int(rate * duration))
    x = np.sin(f * 2. * np.pi * t)
    display(Audio(x, rate=rate, autoplay=True))

notes = 'C,C#,D,D#,E,F,F#,G,G#,A,A#,B,C2,C#2,D2,D#2'.split(',')
freqs = 440. * 2**(np.arange(3, 3 + len(notes)) / 12.)
notes = {name:freq for name, freq in list(zip(notes, freqs))}
jupyter_beep = lambda: synth(notes['F'], 2.0)
