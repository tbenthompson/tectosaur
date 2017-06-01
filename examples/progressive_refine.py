

def refined_free_surface():
    w = 10
    minsize = 0.02
    slope = 400
    maxsize = 0.02
    pts = np.array([[-w, -w, 0], [w, -w, 0], [w, w, 0], [-w, w, 0]])
    tris = np.array([[0, 1, 3], [3, 1, 2]])

    addedpts = 4
    it = 0
    while addedpts > 0:
        pts, tris = mesh.remove_duplicate_pts((pts, tris))
        print(it, addedpts)
        it += 1
        newpts = pts.tolist()
        newtris = []
        addedpts = 0
        # size = np.linalg.norm(np.cross(
        #     pts[tris[:, 1]] - pts[tris[:, 0]],
        #     pts[tris[:, 2]] - pts[tris[:, 0]]
        # ), axis = 1) / 2.0
        # centroid = np.mean(pts[tris], axis = 2)
        # r2 = np.sum(centroid ** 2, axis = 1)
        for i, t in enumerate(tris):
            size = np.linalg.norm(np.cross(
                pts[t[1]] - pts[t[0]],
                pts[t[2]] - pts[t[0]]
            )) / 2.0
            centroid = np.mean(pts[t], axis = 0)
            A = (centroid[0] / 1.5) ** 2 + (centroid[1] / 1.0) ** 2
            # A = np.sum(centroid ** 2)
            if (A < slope * size and size > minsize) or size > maxsize:
            # print(r2[i], size[i])
            # if r2[i] < size[i] and size[i] > minsize:
                newidx = len(newpts)
                newpts.extend([
                    (pts[t[0]] + pts[t[1]]) / 2,
                    (pts[t[1]] + pts[t[2]]) / 2,
                    (pts[t[2]] + pts[t[0]]) / 2
                ])
                newtris.extend([
                    [t[0], newidx, newidx + 2],
                    [newidx, t[1], newidx + 1],
                    [newidx + 1, t[2], newidx + 2],
                    [newidx + 1, newidx + 2, newidx]
                ])
                addedpts += 3
            else:
                newtris.append(t)
        pts = np.array(newpts)
        tris = np.array(newtris)
    final_tris = scipy.spatial.Delaunay(np.array([pts[:,0],pts[:,1]]).T).simplices
    plt.triplot(pts[:, 0], pts[:, 1], final_tris, linewidth = 0.5)
    plt.show()
    print('npts: ' + str(pts.shape[0]))
    print('ntris: ' + str(final_tris.shape[0]))
    return pts, final_tris
