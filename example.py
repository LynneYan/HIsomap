from HIsomap import HIsomap
import numpy as np
import sys
import os
import sklearn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot_original_data_in_3d(X, links, Landmark, color, BP, title, show_skeleton="off"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=3, c=color,
               cmap=plt.cm.get_cmap('coolwarm'))
    if  show_skeleton == "on":
        ax.scatter(Landmark[:, 0], Landmark[:, 1], Landmark[:, 2], cmap=plt.cm.Spectral,
                   c='black', s=100,  marker=(5, 1), label='Landmarks(Centroid)')
        if len(BP) > 0:
            base_point = BP
            ax.scatter(base_point[0], base_point[1], base_point[2],
                       cmap=plt.cm.Spectral, c='red', s=250,  marker=(5, 1), label='Base Point')

        for i in range(0, len(links)):
            ax.plot([Landmark[links[i][0]][0], Landmark[links[i][1]][0]], [Landmark[links[i][0]][1],
                                                                       Landmark[links[i][1]][1]], [Landmark[links[i][0]][2], Landmark[links[i][1]][2]], color='black')
        ax.legend()
    plt.title("Dataset: " + title)
    plt.show()

def plot_projection(Y, proj_landmark, links, color, title, show_skeleton="off"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Y[:, 0].real, Y[:, 1].real, s=3, c=color,
               cmap=plt.cm.get_cmap('coolwarm'))
    if  show_skeleton == "on":
        ax.scatter(proj_landmark[:, 0], proj_landmark[:, 1], c='black', s=100,  marker=(
        5, 1), label='Landmarks(Homology-Preserving)')
        link_num = len(links)
        for i in range(0, link_num):
            ax.plot([proj_landmark[links[i][0]][0], proj_landmark[links[i][1]][0]], [
                proj_landmark[links[i][0]][1], proj_landmark[links[i][1]][1]], color='black')
        ax.legend()
    plt.title("Dataset: " + title)
    plt.show()



data_dir = './data'
data_types = ['SwissHole', 'octa', 'FishingNet', '4elt', 'Mice', 'Portraits']

nr_cubess = [25, 20, 30, 10, 12, 4]
ps = [0.2, 0.2, 0.5, 0.1, 0.2, 0.4]
epss = [0.8, 150, 0.4, 2, 5, 8]
minPts = [5, 5, 5, 5, 10, 2]
BPs = ['EP', 'BC', 'EP', 'EP', 'EP', 'EP']
auto_tunings = ['on', 'off', 'off', 'off', 'off', 'off']

for i in range(0,len(data_types)):
    data_type = data_types[i]
    nr_cubes = nr_cubess[i]
    p = ps[i]
    eps = epss[i]
    minPt = minPts[i]
    BP = BPs[i]
    auto_tuning = auto_tunings[i]
    
    file_name = os.path.join(data_dir, data_type + '.txt')
    X = np.loadtxt(file_name)
    if sys.version_info >= (3, 0):
        proj = HIsomap.HIsomap(nr_cubes=nr_cubes, overlap_perc=p, clusterer=sklearn.cluster.DBSCAN(eps=eps, min_samples=minPt), filter_function="base_point_geodesic_distance", BP=BP, auto_tuning=auto_tuning)
    else:
        proj = HIsomap(nr_cubes=nr_cubes, overlap_perc=p, clusterer=sklearn.cluster.DBSCAN(eps=eps, min_samples=minPt), filter_function="base_point_geodesic_distance", BP=BP, auto_tuning=auto_tuning)
    Y = proj.fit_transform(X)

    Landmark = proj.get_skeleton_nodes()
    links = proj.get_skeleton_links()
    BP = proj.get_base_point()
    color = proj.get_scalar_value()
    landmark_index = proj.get_landmark_index()

    plot_original_data_in_3d(X, links, Landmark, color, BP, data_type, show_skeleton="on")
    plot_projection(Y, Y[landmark_index], links, color, data_type, show_skeleton="off")

