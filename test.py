from HIsomap import HIsomap
import numpy as np
import sys

X = np.loadtxt('X.txt')
if sys.version_info >= (3, 0):
    proj = HIsomap.HIsomap(nr_cubes=25, overlap_perc=0.1, show_skeleton="on", show_projection="on",
                   filter_function="base_point_geodesic_distance", auto_tuning='on')
else:
    proj = HIsomap(nr_cubes=25, overlap_perc=0.1, show_skeleton="on", show_projection="on",
                   filter_function="base_point_geodesic_distance", auto_tuning='on')
Y = proj.fit_transform(X)


