# HIsomap

This is a Python implementation of homology-preserving dimension reduction(DR) algorithm. The implementation is described in ["Homology-Preserving Dimensionality Reduction via Manifold Landmarking and Tearing"](https://arxiv.org/pdf/1806.08460.pdf).

There is also a [demo](https://github.com/LynneYan/HomologyDR_Tearing) for both homology-preserving manifold landmarking and tearing.


- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)


## Installation

Tested with Python 2.7&3.7, MacOS and Linux.

### Dependencies

HIsomap requires:

  - Python (>= 2.7 or >= 3.3)
  - NumPy
  - sklearn

Running examples requires:

  - matplotlib


### Installation 

Python2

```
$ git clone https://github.com/LynneYan/HIsomap.git
$ cd HIsomap
$ sudo python setup.py install
```

Python3

```
$ git clone https://github.com/LynneYan/HIsomap.git
$ cd HIsomap
$ sudo python3 setup.py install
```

### Checking your HIsomap Installation

If you applied all the above steps successfully, you can open terminal and see "HIsomap X.XX" in pip list.

Python2
```
$ pip list
```
Python3

```
$ pip3 list
```

### Run example

Python2
```
$ python example.py
```

Python3
```
$ python3 example.py
```

## Features

```python
class HIsomap(n_components=2, filter_function="base_point_geodesic_distance", BP='EP', nr_cubes=20, 
              overlap_perc=0.2, auto_tuning="off", n_neighbors=8, eigen_solver='auto', n_jobs=1, 
              clusterer=sklearn.cluster.DBSCAN(eps=0.6, min_samples=5))
```
### Parameters

- **n_components**, int, optional, default: 2
  - Number of dimensions in which to immerse the dissimilarities.

- **filter_function**, string, optional, default: "base_point_geodesic_distance"
  - A string from ["sum", "mean", "median", "max", "min", "std", "dist_mean", "l2norm", "knn_distance_n", "height", "width", "base_point_geodesic_distance", "dist_mean", "eccentricity", "Guass_density", "density_estimator", "integral_geodesic_distance", "graph_Laplacian", "Guass_density_auto"]. 
  - If using knn_distance_n write the number of desired neighbors in place of n: knn_distance_5 for summed distances to 5 nearest neighbors.
  - If using base_point_geodesic_distance, you can adjust the parameter "BP" to locate the base point.

- **BP**, string, optional, default: "EP"
  - A string from ["EP", "BC", "DR"].
  - *EP* means extremal point, *BC* means barycenter, and *DR* means densest region.

- **nr_cubes**, int, optional, default: 20
  - The number of intervals/hypercubes to create.

- **overlap_perc**, float, optional, default: 0.2
  - The percentage of overlap "between" the intervals/hypercubes.

- **auto_tuning**, string, optional, default: "off"
  - A string from ["off", "on"].
  - If "off", the input data will be divided into nr_cube cubes with fixed length of interval.
  - If "on", the input data will be divided into nr_cube cubes where each cube contain roughly the same number of points. In this case, the lengths of intervals are different. This means, in dense region, there will be more number of cubes than other regions when auto_tuning is "on".

- **n_neighbors**, int, optional, default: 8
  - Number of neighbors to consider for each point in Isomap.

- **eigen_solver**, string, optional, default: "auto"
  - A string from ["auto", "arpack", "dense"].
  - *auto*: Attempt to choose the most efficient solver for the given problem.
  - *arpack*: Use Arnoldi decomposition to find the eigenvalues and eigenvectors.
  - *dense*: Use a direct solver (i.e. LAPACK) for the eigenvalue decomposition.

- **n_jobs**, int or None, optional, default: 1
  - The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

- **cluster**, algorithm, optional, default: sklearn.cluster.DBSCAN(eps=0.6, min_samples=5)
  - Scikit-learn API compatible clustering algorithm. Must provide `fit` and `predict`.

```python
__init__(self, n_components=2, filter_function="base_point_geodesic_distance", BP='EP', nr_cubes=20, 
         overlap_perc=0.2, auto_tuning="off", n_neighbors=8, eigen_solver='auto', n_jobs=1, 
         clusterer=sklearn.cluster.DBSCAN(eps=0.6, min_samples=5))
```
Initialize self. 

```python
fit_transform(self, X, y=None, init=None)
```
Fit the data from X, and returns the embedded coordinates.

### Parameters
- **X**, array, shape (n_samples, n_features).
  - Input data. 
- **y**, Ignored

- **init**, ndarray, shape (n_samples,), optional, default: None
  - Starting configuration of the embedding to initialize the SMACOF algorithm. By default, the algorithm is initialized with a randomly chosen array.

### Returns
- **Y**, array, shape (n_samples, n_components)
  - Projected output.

```python
get_landmark_index(self)
```
### Returns
- **landmarks_indexes**, int list
  - The indexes of landmarks in input data.

```python
get_skeleton_nodes(self)
```
### Returns
- **landmarks**, ndarray, shape (n_landmarks, n_features).
  - Nodes of mapper graph in original domain.


```python
get_skeleton_links(self)
```
### Returns
- **skeleton**, ndarray, shape (n_link, 2).
  - Edge of mapper graph.


## Usage

## Citation

## License