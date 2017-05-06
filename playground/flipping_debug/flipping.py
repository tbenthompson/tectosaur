# import numpy as np
# from tectosaur.elastic import T
# tensor = [[T(i,j) for j in range(3)] for i in range(3)]
# n = np.array([1, 0, 0])
# srcpt = [1.0, 0.0, 0.0]
# n *= 1
# params = {'nu': 0.25, 'xx': 0, 'xy': 0, 'xz': 0, 'yx': srcpt[0], 'yy': srcpt[1], 'yz': srcpt[2], 'lx': n[0], 'ly': n[1], 'lz': n[2]}
# result = np.array([[float(tensor[i][j]['expr'].subs(params).evalf()) for j in range(3)] for i in range(3)])
# print(result)


from tectosaur_tables.coincident import eval_tri_integral, make_coincident_params
p = make_coincident_params('H', 1e-7, 25, True, True, 25, 25, 1e-4, 3, False, 1, 1, 1)

tri = [[0,0,0],[1,0,0],[0.5,0.5,0.0]]
I = eval_tri_integral(tri, 0.25, p)[:,0].reshape((3,3,3,3))

rot = [2,1,0]
I2_rot = eval_tri_integral([tri[rot[0]], tri[rot[1]], tri[rot[2]]], 0.25, p)[:,0]
I2 = I2_rot.reshape((3,3,3,3))[rot,:,:,:][:,:,rot,:]
import ipdb; ipdb.set_trace()
