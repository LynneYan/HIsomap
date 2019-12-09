from HIsomap import HIsomap
import numpy as np

X = np.loadtxt('X.txt')
proj = HIsomap(nr_cubes=20, show_skeleton="on", show_projection="on", filter_function="base_point_geodesic_distance")
Y = proj.fit_transform(X)

