#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:45:29 2021

@author: zfd297
"""

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import sklearn.manifold._t_sne as ts
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.io import mmread
from time import time
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils import check_random_state
from sklearn.manifold import _barnes_hut_tsne
from scipy import linalg




def calculate_high_P(pca_mat, perplexity=30, metric = 'euclidean',n_jobs=None,verbose=0,square_distances='legacy'):
    
    
    knn = NearestNeighbors(algorithm='auto',
                                   n_jobs=None,
                                   metric=metric)
    
    n_samples = pca_mat.shape[0]
    print(n_samples)
    n_neighbors = min(n_samples - 1, int(3. * perplexity + 1))
    t0 = time()
    knn.fit(pca_mat)
    duration = time() - t0
    if verbose:
        print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
            n_samples, duration))

    t0 = time()
    distances_nn = knn.kneighbors_graph(mode='distance')
    duration = time() - t0
    if verbose:
        print("[t-SNE] Computed neighbors for {} samples "
              "in {:.3f}s...".format(n_samples, duration))

    # Free the memory used by the ball_tree
    del knn

    if square_distances is True or metric == "euclidean":
        # knn return the euclidean distance but we need it squared
        # to be consistent with the 'exact' method. Note that the
        # the method was derived using the euclidean method as in the
        # input space. Not sure of the implication of using a different
        # metric.
        distances_nn.data **= 2

    # compute the joint probability distribution for the input space
    P = ts._joint_probabilities_nn(distances_nn, perplexity,
                                verbose)
    
    return P


def ts_kl_divergence_bh(params, P, degrees_of_freedom, n_samples, n_components,
                      angle=0.5, skip_num_points=0, verbose=False,
                      compute_error=True, num_threads=1):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.
    """
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(val_P, X_embedded, neighbors, indptr,
                                      grad, angle, n_components, verbose,
                                      dof=degrees_of_freedom,
                                      compute_error=compute_error,
                                      num_threads=num_threads)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad


def ts_gradient_descent(objective, p0, it, n_iter,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
    
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
        
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it
    tic = time()
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs['compute_error'] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

    return p, error, i
    
def run_tsne(X, P,n_components=2, early_exaggeration=12, n_iter = 1000):
    
    
    n_samples = X.shape[0]
   
    X_embedded = np.random.normal(loc = 0, scale = 1, size = (n_samples, n_components))
    
    degrees_of_freedom = max(n_components - 1, 1)
    opt_args = {
            "it": 0,
            "n_iter_check": 50,
            "min_grad_norm": 1e-7,
            "learning_rate": 200,
            "verbose": 0,
            "kwargs": dict(skip_num_points=0),
            "args": [P, degrees_of_freedom, n_samples, n_components],
            "n_iter_without_progress": 250,
            "n_iter": 250,
            "momentum": 0.5,
        }
    
    opt_args['kwargs']['angle'] = 0.5
    opt_args['kwargs']['verbose'] = 0
    opt_args['kwargs']['num_threads'] = _openmp_effective_n_threads()
    
    params = X_embedded.ravel()
    P *= early_exaggeration
    params, kl_divergence, it = ts_gradient_descent(ts_kl_divergence_bh, params,
                                                      **opt_args)
    
    _EXPLORATION_N_ITER = 250
    P /= early_exaggeration
    remaining = n_iter - _EXPLORATION_N_ITER
    if it < _EXPLORATION_N_ITER or remaining > 0:
        opt_args['n_iter'] = n_iter
        opt_args['it'] = it + 1
        opt_args['momentum'] = 0.8
        opt_args['n_iter_without_progress'] = 250
        params, kl_divergence, it = ts_gradient_descent(ts_kl_divergence_bh, params,
                                                      **opt_args)

    # Save the final number of iterations
    n_iter_ = it
    
    X_embedded = params.reshape(n_samples, n_components)
        
        
    return X_embedded




    

# count_mat = mmread('normalized_data.mtx')
# count_mat = count_mat.A
# count_mat = count_mat.T
# pca = PCA(n_components=30)
# pca_mat = pca.fit_transform(count_mat)


# P = calculate_high_P(pca_mat,perplexity=50)
# re = run_tsne(pca_mat, P)


# #############3
# cell_id = pd.read_csv("right_oder_cellid.csv")
# cell_id = cell_id['0']

# unique_cell_id = np.unique(cell_id)

# cell_id_num = cell_id.copy()

# dict_id_num = dict(zip(cell_id, cell_id_num))

# for i in range(len(unique_cell_id)):
#     tmp = unique_cell_id[i]
#     index = np.where(cell_id == tmp)[0]
#     cell_id_num[index] = i
    

# umaps = pd.read_csv("umap.txt",sep=" ")

# colors = ["#FF8C00", "#32CD32", "#191970", "#00FA9A", "#7B68EE", "#FFD700", "#F08080", 
#           "#FF0000", "#C71585", "#ffff14", "#00FFFF", "#680018", "#caa0ff"]

# color_map = np.array(cell_id.copy())

# for i in range(len(unique_cell_id)):
#     index = np.where(cell_id == unique_cell_id[i])[0]
#     color_map[index] = colors[i]
# #############

# re = run_tsne(pca_mat, P)



# P = calculate_high_P(pca_mat,perplexity=20)
# re = run_tsne(pca_mat, P)

# plt.figure(None,(20,15))

# for i in range(len(unique_cell_id)):
    
#     tmp = np.where(cell_id == unique_cell_id[i])[0]
#     plt.scatter(re[tmp,0], re[tmp,1],  s=50, label=unique_cell_id[i],c = colors[i]  )
# plt.legend(markerscale=2, ncol= 4,prop={'size': 14})
# plt.axis("off")


