from __future__ import division

import km

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path

import json
import numpy as np
import KernelPCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean



class HIsomap(BaseEstimator):
    def __init__(self, n_components=2, filter_function="base_point_geodesic_distance", BP='EP', nr_cubes=20, overlap_perc=0.2, auto_tuning="off", n_neighbors=8, eigen_solver='auto', n_jobs=1, clusterer=sklearn.cluster.DBSCAN(eps=0.6, min_samples=5)):
        self.n_components = n_components
        self.filter_function = filter_function
        self.nr_cubes = nr_cubes
        self.overlap_perc = overlap_perc
        self.clusterer = clusterer
        self.BP = BP
        self.n_neighbors = n_neighbors
        self.eigen_solver = eigen_solver
        self.n_jobs = n_jobs
        self.landmarks_indexes = []
        self.landmarks = []
        self.skeleton = []
        self.basePoint = []
        self.auto_tuning = auto_tuning
        self.lens = []


    @property
    def fit(self, X, y=None, init=None):
        self.fit_transform(X, init=init)
        return self

    def _get_landmarks(self, complex):
        json_s = {}
        json_s["links"] = []
        json_s["nodes"] = []
        k2e = {}  # a key to incremental int dict, used for id's when linking
        for e, k in enumerate(complex["nodes"]):
            children = {}
            children["name"] = []
            for ch in complex["nodes"][k]:
                children["name"].append({"name": ch})
            json_s["nodes"].append(
                {"id": e, "group": e, "children": children["name"]})
            k2e[k] = e
        for k in complex["links"]:
            for link in complex["links"][k]:
                json_s["links"].append({"source": k2e[k], "target": k2e[link]})
        self.landmarks = json_s
        return json_s

    def _compute_skeleton(self, data, X):
        Landmark = []
        node_num = len(data['nodes'])
        for i in range(0, node_num):
            children_num = len(data['nodes'][i]['children'])
            children = []
            for j in range(0, children_num):
                children.append(
                    X[int(data['nodes'][i]['children'][j]['name'])])
                #color[int(data['nodes'][i]['children'][j]['name'])] = i
            children = np.array(children)
            kmeans = KMeans(n_clusters=1).fit(children)
            ck_km = kmeans.cluster_centers_[0]
            dist_min = 10000000
            for i in range(0, children_num):
                if euclidean(children[i], ck_km) < dist_min:
                    dist_min = euclidean(children[i], ck_km)
                    center = children[i]
            Landmark.append(center)
        Landmark = np.array(Landmark)
        links = []
        link_num = len(data['links'])
        for k in range(0, link_num):
            links.append((int(data['links'][k]['source']),
                          int(data['links'][k]['target'])))
        return Landmark, links

    def Landmark_Isomap(self, D, ndims, landmarks, dist_matrix):
        ALL_matrix = D
        if len(dist_matrix) > 0:
            dist_matrix_ = dist_matrix
        else:
            nbrs_ = NearestNeighbors(
                n_neighbors=self.n_neighbors, algorithm=self.eigen_solver, n_jobs=self.n_jobs)
            nbrs_.fit(ALL_matrix)
            kng = kneighbors_graph(
                nbrs_, self.n_neighbors, mode='distance', n_jobs=self.n_jobs)
            dist_matrix_ = graph_shortest_path(kng,
                                               method='auto',
                                               directed=False)
        G_D = dist_matrix_[landmarks, :]
        landmarks = np.array(landmarks)
        G_ = dist_matrix_[landmarks[:, None], landmarks]

        G = G_ ** 2
        G *= -0.5
        n = len(landmarks)
        N = len(D)

        eigenxy, eigenval = KernelPCA.KernelPCA(n_components=ndims,
                                                kernel="precomputed",
                                                eigen_solver='auto',
                                                tol=0, max_iter=None,
                                                n_jobs=self.n_jobs).fit_transform(G)

        xy = eigenxy
        val = eigenval

        for i in range(0, ndims):
            xy[:, i] = xy[:, i]*np.sqrt(val[i])

        xy1 = np.zeros((len(D), ndims))
        LT = xy.transpose()

        for i in range(0, ndims):
            LT[i, :] = LT[i, :]/val[i]
            deltan = G.mean(0)

        for x in range(0, len(D)):
            deltax = G_D[:, x]
            xy1[x, :] = 1/2 * (LT.dot((deltan-deltax))).transpose()

        return xy1, xy1[landmarks]


    def get_landmark_index(self):
        if len(self.landmarks_indexes) == 0:
            print ("Warning: Please run HIsomap.fit_transform() first")
        return self.landmarks_indexes

    def get_skeleton_nodes(self):
        if len(self.landmarks_indexes) == 0:
            print ("Warning: Please run HIsomap.fit_transform() first")
        return self.landmarks

    def get_scalar_value(self):
        if len(self.landmarks_indexes) == 0:
            print ("Warning: Please run HIsomap.fit_transform() first")
        return self.lens.flatten()

    def get_base_point(self):
        if len(self.landmarks_indexes) == 0:
            print ("Warning: Please run HIsomap.fit_transform() first")
        return self.basePoint

    def get_skeleton_links(self):
        if len(self.landmarks_indexes) == 0:
            print ("Warning: Please run HIsomap.fit_transform() first")
        return self.skeleton

    def fit_transform(self, X, y=None, init=None):
        mapper = km.KeplerMapper()
        dist_matrix, lens, bp = mapper.fit_transform(
            X, projection=self.filter_function, BP=self.BP)
        graph = mapper.map(lens, X,
                           clusterer=self.clusterer,
                           nr_cubes=self.nr_cubes,
                           overlap_perc=self.overlap_perc,
                           auto_tuning=self.auto_tuning)

        landmarks = self._get_landmarks(graph)
        Landmark, links = self._compute_skeleton(landmarks, X)

        self.landmarks = Landmark
        self.skeleton = links
        self.basePoint = bp
        self.lens = lens

        X = X.tolist()
        Index_of_landmarks = []
        for i in range(0, len(Landmark)):
            Index_of_landmarks.append((X.index(Landmark.tolist()[i])))

        self.landmarks_indexes = Index_of_landmarks
        Y, proj_landmark = self.Landmark_Isomap(
            X, 2, Index_of_landmarks, dist_matrix)


        return Y
