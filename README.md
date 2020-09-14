(NOTE: Unmaintained)


Observe the tectonosaurus and the elastosaurus romp pleasantly through the fields of stress.


Tectosaur is an implementation of the elastic boundary element method, oriented towards problems involving faults. It can certainly be used for non-fault elasticity problems, too! Tectosaur is built on a new numerical integration methodology for computing Green's function integrals. This allows a great deal of flexibility in the problems it can solve. The use of  efficient algorithms like the Fast Multipole Method and the use of parallelization and GPU acceleration lead to very rapid solution of very large problems. To summarize the practical capabilities of tectosaur:

1. Solving complex geometric static elastic boundary value problems including

* topography
* earth curvature
* material property contrasts

2. Problems with millions of elements can be solved in minutes on a desktop computer. 
3. No need for volumetric meshing is ideal for problems where the fault is the topic of interest.
4. Rapid model iteration

Further documentation and examples will absolutely be available in the future! Until then, however, tectosaur will be rapidly changing and developing and any users should expect little to no support and frequent API breaking changes. 

# Installation instructions

1. Tectosaur requires Python 3.5 or greater. 
2. You will need to have either PyCUDA or PyOpenCL installed. If you have an NVidia GPU, install PyCUDA for best performance. Try running `pip install pycuda`. If that fails, you can follow the [more detailed instructions on the PyCUDA wiki](https://wiki.tiker.net/PyCuda/Installation). And [here are detailed instructions for installing PyOpenCL](https://documen.tician.de/pyopencl/misc.html).
3. Install numpy: `pip install numpy`
4. `git clone https://github.com/tbenthompson/tectosaur.git`
5. `git checkout dissertation_stable` if you want the somewhat stable version used for my dissertation.
5. Enter that directory and run `pip install .`

# Running the examples

1. Check that Jupyter is installed!
2. Launch a Jupyter notebook or lab server. 
3. Navigate to `tectosaur/examples/notebooks`.
4. Open and run the examples!
