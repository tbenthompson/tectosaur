
import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')
res = adaptive_integrate.integrate_interior(
    "U", [0,0,1],[0,0,1],
    [[0,0,0],[1,0,0],[0,1,0]],
    1e-2, 1.0, 0.25
)
print(res)
