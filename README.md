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

**n_components**, int, optional, default: 2
    Number of dimensions in which to immerse the dissimilarities.
**filter_function**, str, optional, default: "base_point_geodesic_distance"
    Filter function. A string from ["sum", "mean", "median", "max", "min", "std", "dist_mean", "l2norm", "knn_distance_n"]. If using knn_distance_n write the number of desired neighbors in place of n: knn_distance_5 for summed distances to 5 nearest neighbors.

## Usage

## Citation

## License