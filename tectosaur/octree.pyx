import numpy as np
cimport numpy as np

cdef class Octree:
    cdef public np.ndarray center;
    cdef public np.ndarray half_width;
    cdef public np.ndarray pts;
    cdef public list children;

    def __init__(self, int max_pts, np.ndarray pts):
        self.construct(max_pts, pts)

    cdef construct(Octree self, int max_pts, np.ndarray[double,ndim=2] pts):
        cdef int s1
        cdef int s2
        cdef int s3
        cdef np.ndarray[double,ndim=1] min_corner
        cdef np.ndarray[double,ndim=1] max_corner

        if pts.shape[0] == 0:
            self.pts = None
            self.children = None
            return

        min_corner = np.min(pts, axis = 0)
        max_corner = np.max(pts, axis = 0)
        self.center = (min_corner + max_corner) / 2
        self.half_width = (min_corner - max_corner) / 2

        if pts.shape[0] <= max_pts:
            self.pts = pts
            self.children = None
        else:
            self.pts = None
            self.children = []
            for s1 in range(2):
                for s2 in range(2):
                    for s3 in range(2):
                        self.build_child(s1, s2, s3, max_pts, pts)

    cdef build_child(self, int s1, int s2, int s3, int max_pts, np.ndarray pts):
        cdef np.ndarray cond
        cdef np.ndarray child_pts

        if s1 == 0:
            cond = pts[:,0] < self.center[0]
        else:
            cond = pts[:,0] >= self.center[0]

        if s2 == 0:
            cond = np.logical_and(cond, pts[:,1] < self.center[1])
        else:
            cond = np.logical_and(cond, pts[:,1] >= self.center[1])

        if s3 == 0:
            cond = np.logical_and(cond, pts[:,2] < self.center[2])
        else:
            cond = np.logical_and(cond, pts[:,2] >= self.center[2])

        child_pts = pts[cond]
        self.children.append(Octree(max_pts, child_pts))
