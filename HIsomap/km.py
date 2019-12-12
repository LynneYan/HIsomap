from __future__ import division
import progressreporter
import numpy as np
from collections import defaultdict
import json
import itertools
from sklearn import cluster, preprocessing, manifold, decomposition
from scipy.spatial import distance
from datetime import datetime
import sys
import inspect
import json
import random
from scipy import stats
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, euclidean
import math
import scipy.sparse
import scipy.sparse.linalg as spla
from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial.distance import squareform, cdist, pdist
if sys.hexversion < 0x03000000:
    from itertools import izip as zip
    range = xrange

#from mapper import n_obs, cmappertoolserror


class KeplerMapper(object):

    def __init__(self, verbose=-1):
        self.verbose = verbose
        self.chunk_dist = []
        self.overlap_dist = []
        self.d = []
        self.nr_cubes = 0
        self.overlap_perc = 0
        self.clusterer = False

    def fit_transform(self, X, projection="sum", scaler=preprocessing.MinMaxScaler(), distance_matrix=False):

        self.inverse = X
        self.scaler = scaler
        self.projection = str(projection)
        self.distance_matrix = distance_matrix

        if self.distance_matrix in ["braycurtis",
                                    "canberra",
                                    "chebyshev",
                                    "cityblock",
                                    "correlation",
                                    "cosine",
                                    "dice",
                                    "euclidean",
                                    "hamming",
                                    "jaccard",
                                    "kulsinski",
                                    "mahalanobis",
                                    "matching",
                                    "minkowski",
                                    "rogerstanimoto",
                                    "russellrao",
                                    "seuclidean",
                                    "sokalmichener",
                                    "sokalsneath",
                                    "sqeuclidean",
                                    "yule"]:
            X = distance.squareform(distance.pdist(X, metric=distance_matrix))
            if self.verbose > 0:
                print("Created distance matrix, shape: %s, with distance metric `%s`" % (
                    X.shape, distance_matrix))

        # Detect if projection is a class (for scikit-learn)
        try:
            p = projection.get_params()
            reducer = projection
            if self.verbose > 0:
                try:
                    projection.set_params(**{"verbose": self.verbose})
                except:
                    pass
                print("\n..Projecting data using: \n\t%s\n" % str(projection))
            X = reducer.fit_transform(X)
        except:
            pass

        BP = []
        distMatrix = []
        # Detect if projection is a string (for standard functions)
        if isinstance(projection, str):
            if self.verbose > 0:
                print("\n..Projecting data using: %s" % (projection))
            # Stats lenses
            if projection == "sum":  # sum of row
                X = np.sum(X, axis=1).reshape((X.shape[0], 1))
            if projection == "mean":  # mean of row
                X = np.mean(X, axis=1).reshape((X.shape[0], 1))
            if projection == "median":  # mean of row
                X = np.median(X, axis=1).reshape((X.shape[0], 1))
            if projection == "max":  # max of row
                X = np.max(X, axis=1).reshape((X.shape[0], 1))
            if projection == "min":  # min of row
                X = np.min(X, axis=1).reshape((X.shape[0], 1))
            if projection == "std":  # std of row
                X = np.std(X, axis=1).reshape((X.shape[0], 1))
            if projection == "l2norm":
                X = np.linalg.norm(X, axis=1).reshape((X.shape[0], 1))
            if projection == "height":  # std of row
                height = X[:, 2].reshape((X.shape[0], 1))
                X = height
            if projection == "width":  # std of row
                height = X[:, 0].reshape((X.shape[0], 1))
                X = height
            if projection == "base_point_geodesic_distance":
                distMatrix, X, BP = Base_Point_Geodesic_Distance(X, 8)
                X = X.reshape((X.shape[0], 1))

            if projection == "dist_mean":  # Distance of x to mean of X
                X_mean = np.mean(X, axis=0)
                X = np.sum(np.sqrt((X - X_mean)**2),
                           axis=1).reshape((X.shape[0], 1))

            if projection == "eccentricity":
                X = eccentricity(X, 1, {}, None).reshape((X.shape[0], 1))
            if projection == "Guass_density":
                #sigma=float(raw_input("Please enter sigma: "))
                sigma = 0.8
                X = Gauss_density(X, sigma, {}, None).reshape((X.shape[0], 1))

            if projection == "density_estimator":
                k = 15
                X = eccentricity(X, 15).reshape((X.shape[0], 1))

            if projection == "integral_geodesic_distance":
                X = Integral_Geodesic_Distance(X, 10).reshape((X.shape[0], 1))

            if projection == "graph_Laplacian":
                #eps = float(raw_input("Please enter espion: "))
                eps = 0.2
                X = eccentricity(X, eps).reshape((X.shape[0], 1))

            if projection == "Guass_density_auto":
                kde = stats.gaussian_kde(X.T)
                X = kde(X.T).reshape((X.shape[0], 1))

            if "knn_distance_" in projection:
                n_neighbors = int(projection.split("_")[2])
                if self.distance_matrix:  # We use the distance matrix for finding neighbors
                    X = np.sum(np.sort(X, axis=1)[:, :n_neighbors], axis=1).reshape(
                        (X.shape[0], 1))
                else:
                    from sklearn import neighbors
                    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
                    nn.fit(X)
                    X = np.sum(nn.kneighbors(X, n_neighbors=n_neighbors, return_distance=True)[
                               0], axis=1).reshape((X.shape[0], 1))

        # Detect if projection is a list (with dimension indices)
        if isinstance(projection, list):
            if self.verbose > 0:
                print("\n..Projecting data using: %s" % (str(projection)))
            X = X[:, np.array(projection)]

        # Scaling
        if scaler is not None:
            if self.verbose > 0:
                print("\n..Scaling with: %s\n" % str(scaler))
            X = scaler.fit_transform(X)

        return distMatrix, X, BP

    def map(self, projected_X, inverse_X=None, clusterer=cluster.DBSCAN(eps=0.5, min_samples=3), nr_cubes=10, overlap_perc=0.1):
        # This maps the data to a simplicial complex. Returns a dictionary with nodes and links.
        #
        # Input:    projected_X. A Numpy array with the projection/lens.
        # Output:    complex. A dictionary with "nodes", "links" and "meta information"
        #
        # parameters
        # ----------
        # projected_X  	projected_X. A Numpy array with the projection/lens. Required.
        # inverse_X    	Numpy array or None. If None then the projection itself is used for clustering.
        # clusterer    	Scikit-learn API compatible clustering algorithm. Default: DBSCAN
        # nr_cubes    	Int. The number of intervals/hypercubes to create.
        # overlap_perc  Float. The percentage of overlap "between" the intervals/hypercubes.

        start = datetime.now()

        # Helper function
        def cube_coordinates_all(nr_cubes, nr_dimensions):
            # Helper function to get origin coordinates for our intervals/hypercubes
            # Useful for looping no matter the number of cubes or dimensions
            # Example:   	if there are 4 cubes per dimension and 3 dimensions
            #       		return the bottom left (origin) coordinates of 64 hypercubes,
            #       		as a sorted list of Numpy arrays
            # TODO: elegance-ify...
            l = []
            for x in range(nr_cubes):
                l += [x] * nr_dimensions
            return [np.array(list(f)) for f in sorted(set(itertools.permutations(l, nr_dimensions)))]

        nodes = defaultdict(list)
        links = defaultdict(list)
        meta = defaultdict(list)
        graph = {}
        self.nr_cubes = nr_cubes
        self.clusterer = clusterer
        self.overlap_perc = overlap_perc

        # If inverse image is not provided, we use the projection as the inverse image (suffer projection loss)
        if inverse_X is None:
            inverse_X = projected_X

        if self.verbose > 0:
            print("Mapping on data shaped %s using lens shaped %s\n" %
                  (str(inverse_X.shape), str(projected_X.shape)))

        # We chop up the min-max column ranges into 'nr_cubes' parts
        self.chunk_dist = (np.max(projected_X, axis=0) -
                           np.min(projected_X, axis=0))/nr_cubes

        # We calculate the overlapping windows distance
        self.overlap_dist = self.overlap_perc * self.chunk_dist

        # We find our starting point
        self.d = np.min(projected_X, axis=0)

        # Use a dimension index array on the projected X
        # (For now this uses the entire dimensionality, but we keep for experimentation)
        di = np.array([x for x in range(projected_X.shape[1])])

        # Prefix'ing the data with ID's
        ids = np.array([x for x in range(projected_X.shape[0])])
        projected_X = np.c_[ids, projected_X]
        inverse_X = np.c_[ids, inverse_X]

        # Algo's like K-Means, have a set number of clusters. We need this number
        # to adjust for the minimal number of samples inside an interval before
        # we consider clustering or skipping it.
        cluster_params = self.clusterer.get_params()
        try:
            min_cluster_samples = cluster_params["n_clusters"]
        except:
            min_cluster_samples = 1
        if self.verbose > 0:
            print("Minimal points in hypercube before clustering: %d" %
                  (min_cluster_samples))

        # Subdivide the projected data X in intervals/hypercubes with overlap
        if self.verbose > 0:
            total_cubes = len(
                list(cube_coordinates_all(nr_cubes, di.shape[0])))
            print("Creating %s hypercubes." % total_cubes)

        for i, coor in enumerate(cube_coordinates_all(nr_cubes, di.shape[0])):

            # Slice the hypercube
            hypercube = projected_X[np.invert(np.any((projected_X[:, di+1] >= self.d[di] + (coor * self.chunk_dist[di])) &
                                                     (projected_X[:, di+1] < self.d[di] + (coor * self.chunk_dist[di]) + self.chunk_dist[di] + self.overlap_dist[di]) == False, axis=1))]

            if self.verbose > 1:
                print("There are %s points in cube_%s / %s with starting range %s" %
                      (hypercube.shape[0], i, total_cubes, self.d[di] + (coor * self.chunk_dist[di])))

            # If at least min_cluster_samples samples inside the hypercube
            if hypercube.shape[0] >= min_cluster_samples:
                # Cluster the data point(s) in the cube, skipping the id-column
                # Note that we apply clustering on the inverse image (original data samples) that fall inside the cube.
                inverse_x = inverse_X[[int(nn) for nn in hypercube[:, 0]]]

                clusterer.fit(inverse_x[:, 1:])

                if self.verbose > 1:
                    print("Found %s clusters in cube_%s\n" % (
                        np.unique(clusterer.labels_[clusterer.labels_ > -1]).shape[0], i))

                # Now for every (sample id in cube, predicted cluster label)
                for a in np.c_[hypercube[:, 0], clusterer.labels_]:
                    if a[1] != -1:  # if not predicted as noise
                        cluster_id = str(coor[0])+"_"+str(i)+"_"+str(a[1])+"_"+str(coor)+"_"+str(
                            self.d[di] + (coor * self.chunk_dist[di]))  # TODO: de-rudimentary-ify

                        # Append the member id's as integers
                        nodes[cluster_id].append(int(a[0]))
                        meta[cluster_id] = {
                            "size": hypercube.shape[0], "coordinates": coor}
                        size = hypercube.shape[0]

            else:
                if self.verbose > 1:
                    print("Cube_%s is empty.\n" % (i))

        # Create links when clusters from different hypercubes have members with the same sample id.
        candidates = itertools.combinations(nodes.keys(), 2)
        for candidate in candidates:
            # if there are non-unique members in the union
            if len(nodes[candidate[0]]+nodes[candidate[1]]) != len(set(nodes[candidate[0]]+nodes[candidate[1]])):
                links[candidate[0]].append(candidate[1])

        # Reporting
        if self.verbose > 0:
            nr_links = 0
            for k in links:
                nr_links += len(links[k])
            print("\ncreated %s edges and %s nodes in %s." %
                  (nr_links, len(nodes), str(datetime.now()-start)))
        graph["nodes"] = nodes
        graph["links"] = links
        graph["meta_graph"] = self.projection
        graph["meta_nodes"] = meta
        return graph


def eccentricity(data, exponent=1.,  metricpar={}, callback=None):
    if data.ndim == 1:
        assert metricpar == {}, 'No optional parameter is allowed for a dissimilarity matrix.'
        ds = squareform(data, force='tomatrix')
        if exponent in (np.inf, 'Inf', 'inf'):
            return ds.max(axis=0)
        elif exponent == 1.:
            ds = np.power(ds, exponent)
            return ds.sum(axis=0)/float(np.alen(ds))
        else:
            ds = np.power(ds, exponent)
            return np.power(ds.sum(axis=0)/float(np.alen(ds)), 1./exponent)
    else:
        progress = progressreporter.progressreporter(callback)
        N = np.alen(data)
        ecc = np.empty(N)
        if exponent in (np.inf, 'Inf', 'inf'):
            for i in range(N):
                ecc[i] = cdist(data[(i,), :], data, **metricpar).max()
                progress((i+1)*100//N)
        elif exponent == 1.:
            for i in range(N):
                ecc[i] = cdist(data[(i,), :], data, **metricpar).sum()/float(N)
                progress((i+1)*100//N)
        else:
            for i in range(N):
                dsum = np.power(cdist(data[(i,), :], data, **metricpar),
                                exponent).sum()
                ecc[i] = np.power(dsum/float(N), 1./exponent)
                progress((i+1)*100//N)
        return ecc


def Density_Estimator(D, k):
    n_jobs = 1
    nbrs_ = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=n_jobs)
    nbrs_.fit(D)
    kng = kneighbors_graph(nbrs_, k, mode='distance', n_jobs=n_jobs)

    DE = np.zeros(len(D))
    dist_matrix = kng.toarray()
    for i in range(0, len(D)):
        for j in range(0, len(D)):
            if dist_matrix[i][j] > 0:
                DE[i] += dist_matrix[i][j]**2
    for i in range(0, len(D)):
        DE[i] = -1/k*math.sqrt(DE[i])

    return DE


def Base_Point_Geodesic_Distance(D, n_neighbors):
    n_jobs = 1
    ALL_matrix = D
    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors,
                             algorithm='auto', n_jobs=n_jobs)
    nbrs_.fit(ALL_matrix)
    kng = kneighbors_graph(nbrs_, n_neighbors, mode='distance', n_jobs=n_jobs)
    dist_matrix_ = graph_shortest_path(kng, method='auto', directed=False)
    G = dist_matrix_
    Index = np.argmax(G[random.randint(0, len(D)+1)])
    return G, G[Index], D[Index]


def Integral_Geodesic_Distance(D, n_neighbors):
    n_jobs = 1
    ALL_matrix = D
    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors,
                             algorithm='auto', n_jobs=n_jobs)
    nbrs_.fit(ALL_matrix)
    kng = kneighbors_graph(nbrs_, n_neighbors, mode='distance', n_jobs=n_jobs)
    dist_matrix_ = graph_shortest_path(kng, method='auto', directed=False)
    G = dist_matrix_
    IGD = np.zeros(len(G))
    for i in range(0, len(G)):
        IGD[i] = sum(G[i])

    min_ = min(IGD)
    max_ = max(IGD)
    for i in range(0, len(G)):
        IGD[i] = (IGD[i]-min_)/max_
    return IGD


def Gauss_density(data, sigma, metricpar={}, callback=None):
    denom = -2.*sigma*sigma
    if data.ndim == 1:
        assert metricpar == {}, ('No optional parameter is allowed for a '
                                 'dissimilarity matrix.')
        ds = squareform(data, force='tomatrix')
        dd = np.exp(ds*ds/denom)
        dens = dd.sum(axis=0)
    else:
        progress = progressreporter.progressreporter(callback)
        N = np.alen(data)
        dens = np.empty(N)
        for i in range(N):
            d = cdist(data[(i,), :], data, **metricpar)
            dens[i] = np.exp(d*d/denom).sum()
            progress(((i+1)*100//N))
        dens /= N*np.power(np.sqrt(2*np.pi)*sigma, data.shape[1])
    return dens


def graph_Laplacian(data, eps, n=1, k=1, weighted_edges=False, sigma_eps=1.,
                    normalized=True,
                    metricpar={}, verbose=True,
                    callback=None):
    assert n >= 1, 'The rank of the eigenvector must be positive.'
    assert isinstance(k, int)
    assert k >= 1
    if data.ndim == 1:
        # dissimilarity matrix
        assert metricpar == {}, ('No optional parameter is allowed for a '
                                 'dissimilarity matrix.')
        D = data
        N = n_obs(D)
    else:
        # vector data
        D = pdist(data, **metricpar)
        N = len(data)
    if callback:
        callback('Computing: neighborhood graph.')
    rowstart, targets, weights = \
        neighborhood_graph(D, k, eps, diagonal=True,
                           verbose=verbose, callback=callback)

    c = ncomp(rowstart, targets)
    if (c > 1):
        print('The neighborhood graph has {0} components. Return zero values.'.
              format(c))
        return zero_filter(data)

    weights = Laplacian(rowstart, targets, weights, weighted_edges,
                        eps, sigma_eps, normalized)

    L = scipy.sparse.csr_matrix((weights, targets, rowstart))
    del weights, targets, rowstart

    if callback:
        callback('Computing: eigenvectors.')

    assert n < N, ('The rank of the eigenvector must be smaller than the number '
                   'of data points.')

    if hasattr(spla, 'eigsh'):
        w, v = spla.eigsh(L, k=n+1, which='SA')
    else:  # for SciPy < 0.9.0
        w, v = spla.eigen_symmetric(L, k=n+1, which='SA')
    # Strange: computing more eigenvectors seems faster.
    #w, v = spla.eigsh(L, k=n+1, sigma=0., which='LM')
    if verbose:
        print('Eigenvalues: {0}.'.format(w))
    order = np.argsort(w)
    if w[order[0]] < 0 and w[order[1]] < abs(w[order[0]]):
        raise RuntimeError(
            'Negative eigenvalue of the graph Laplacian found: {0}'.format(w))

    return v[:, order[n]]
