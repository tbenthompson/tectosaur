Observe the tectonosaurus and the elastosaurus romp pleasantly through the fields of stress.


Tectosaur is an implementation of the elastic boundary element method, oriented towards problems involving faults. Though it can certainly be used for non-fault elasticity problems, too! Tectosaur is built on a new numerical integration methodology for computing Green's function integrals. This allows a great deal of flexibility in the problems it can solve. The use of  efficient algorithms like the Fast Multipole Method and the use of parallelization and GPU acceleration lead to very rapid solution of very large problems. To summarize the practical capabilities of tectosaur:

1. Solving complex geometric static elastic boundary value problems including

* topography
* earth curvature
* material property contrasts

2. Problems with millions of elements can be solved in minutes on a desktop computer. 
3. No need for volumetric meshing is ideal for problems where the fault is the topic of interest.
4. Rapid model iteration

Further documentation and examples will absolutely be available in the future! Until then, however, tectosaur will be rapidly changing and developing and any users should expect little to no support and frequent API breaking changes. 
