from HIsomap import HIsomap
import numpy as np
import sys

X = np.loadtxt('X.txt')
if sys.version_info >= (3, 0):
    proj = HIsomap.HIsomap(nr_cubes=20, show_skeleton="on", show_projection="on",
                   filter_function="base_point_geodesic_distance")
else:
    proj = HIsomap(nr_cubes=20, show_skeleton="on", show_projection="on",
                   filter_function="base_point_geodesic_distance")
Y = proj.fit_transform(X)
