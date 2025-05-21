---
title: 'DiRe - JAX: A JAX based Dimensionality Reduction Algorithm for Large-scale Data'
tags:
  - Python
  - JAX
  - dimensionality reduction
  - machine learning
  - persistence homology
  - data visualization
authors:
  - name: Alexander Kolpakov
    orcid: 0000-0002-6764-8894
    affiliation: 1
    equal-contrib: true
    corresponding: true
  - name: Igor Rivin
    orcid: 0000-0001-9302-2169
    affiliation: 2
    equal-contrib: true
affiliations:
  - name: University of Austin, Austin TX, USA; akolpakov@uaustin.org
    index: 1
  - name: Temple University, Philadelphia PA, USA; rivin@temple.edu
    index: 2
date: 21 May 2025
bibliography: dire_jax.bib
---

# Summary

DiRe - JAX is a new dimensionality reduction toolkit designed to address some of the challenges faced by traditional methods like UMAP and tSNE 
such as loss of global structure and computational efficiency. Built on the JAX framework, DiRe leverages modern hardware acceleration to provide 
an efficient, scalable, and interpretable solution for visualizing complex data structures and for quantitative analysis of lower-dimensional embeddings. 
The toolkit shows considerable promise in preserving both local and global structures within the data as compared to state-of-the-art UMAP and tSNE implementations. 
This makes it suitable for a wide range of applications in machine learning, bioinformatics, and data science.

# Statement of need

Traditional dimensionality reduction techniques such as UMAP and tSNE are widely used for visualizing high-dimensional data in lower-dimensional spaces, 
usually 2D and sometimes 3D. Other uses include dimensionality reduction to other, possibly higher and thus non-visual dimensions, for the subsequent use of classifiers such as SVMs.

However, these methods often struggle with scalability, interpretability, and preservation of global data structures. UMAP, while fast and scalable, may overemphasize 
local structures at the expense of global data relationships [@pachter]. And tSNE, while known for producing high-quality visualizations, may be computationally 
expensive and sensitive to hyperparameter tuning [@kobak2019art].

DiRe-JAX addresses these challenges by offering a scalable solution that balances the preservation of both local and global structures. Leveraging the JAX 
framework allows DiRe-JAX to efficiently handle large datasets by utilizing GPU/TPU acceleration, making it significantly faster than CPU-based implementations 
without compromising on the quality of the embeddings.

DiRe-JAX also includes a wealth of metrics for analyzing embedding quality and for hyperparameter tuning. Given its runtime efficiency, tasks such as grid-search 
hyperparameter optimization become feasible even in low-cost environments like Google Colab.

This makes DiRe-JAX an essential toolkit for researchers and practitioners working with complex, high-dimensional data.

# Main methods

The main class of DiRe-JAX is `DiRe`. Let $X \subset \mathbb{R}^n$ be the input data realized as a NumPy array. Then, `DiRe` performs the following main steps:

1. **Capturing dataset topology:** Create the kNN graph of $X$, say $\Gamma$, for a given number of neighbors $k$ (`n_neighbors`) by calling `make_knn_adjacency`. This step uses a JAX kernel specifically developed by the authors to perform the computation on CPU, GPU, or TPU settings. Other libraries like FAISS [@faiss] may also be used, although they do not provide the same hardware universality.
2. **Initial dimension reduction:** Produce $Y \subset \mathbb{R}^d$, the initial embedding of $X$, with $d \ll n$ (usually $d=2$ or $3$) given by the `dimension` parameter, using one of the available embedding methods: `random` (random projections from the Johnson–Lindenstrauss Lemma), `spectral` (using the kNN graph $\Gamma$ to construct the weighted Laplacian, optionally applying a similarity kernel), or `pca` (classical or kernel-based).
3. **Layout optimization:** Call `do_layout` to adjust the lower-dimensional embedding $Y$ to conform to the similarity structure of the higher-dimensional data $X$. This is done via a force-directed layout where the role of “forces” is played by probability kernels (the distributions can be adjusted via parameters `min_dist` and `spread`).

The initial embedding is stored in `self.init_embedding`, while the optimized layout is stored in `self.layout`. Both can be accessed after the main method `fit_transform` is called, for detailed comparison and analysis.

## Random Projection embedding

This embedding is based on the following classical Johnson–Lindenstrauss Lemma [@JL]:

> **Johnson–Lindenstrauss Lemma (Probabilistic Form)**
>
> Given $0 < \epsilon < 1$ and an integer $n$, let $X$ be a set of $n$ points in $\mathbb{R}^d$. For a random linear map $f: \mathbb{R}^d \to \mathbb{R}^k$ where $k = O\left(\frac{\log n}{\epsilon^2}\right)$, with high probability, for all $u, v \in X$:
>
> $$
> (1 - \epsilon) \lVert u - v \rVert^2 \le \lVert f(u) - f(v) \rVert^2 \le (1 + \epsilon) \lVert u - v \rVert^2.
> $$
>
> The value
>
> $$
> \text{dist}(f) = \frac{\lVert f(u) - f(v) \rVert}{\lVert u - v \rVert}
> $$
>
> is called the *distortion* of $f$, and is expected to be close to $1.0$ for a high-quality embedding.

Random projections are simple and computationally inexpensive, but can suffer from cluttered outputs when $k \ll d$ due to variance reduction in the projected data.

## Principal Component Analysis embedding

PCA seeks to preserve as much dataset variance as possible by projecting onto the top $k$ singular vectors. Assume $X$ is column-centered, with covariance matrix $\mathrm{Cov}(X) = \frac{1}{n-1}X^T X$. Compute the singular value decomposition:

$$
X = U \Sigma W^T,
$$

and truncate to the top $k$ singular values $\Sigma_k$ and vectors $W_k$. The rank-$k$ approximation is:

$$
\widehat{X} = U_k \Sigma_k W_k^T,
$$

and the PCA embedding is:

$$
X_k = X W_k.
$$

Under mild conditions on the spectrum, the bottleneck distance between persistence diagrams of $X$ and $X_k$ satisfies [@chazal-persistence]:

$$
d_b(D(X), D(X_k)) \le \lVert \widehat{X} - X \rVert_F = \varepsilon \lVert X \rVert_F.
$$

Thus, PCA approximately preserves topological features up to controlled error.

## Spectral Laplacian embedding

Using the kNN graph $\Gamma$ of $X$, construct the graph Laplacian (optionally with a similarity kernel) and compute the bottom-$k$ eigenvectors for embedding. This method often captures manifold structure but may not preserve global relationships.

## Force-directed layout

After obtaining an initial embedding $Y$, DiRe--JAX applies an iterative force-directed layout to align $Y$’s local structure with that of $X$. Attraction and repulsion forces are modeled after tSNE and UMAP kernels:

$$
\varphi(x) = \frac{1}{1 + a \lVert x \rVert^{2b}},
$$

tuned by `min_dist` $= \delta$ and `spread` $= \sigma$ so that:

- $\varphi(x) \approx 1.0$ for $\lVert x \rVert < \delta$, and
- $\varphi(x) \approx \exp\bigl(-(\lVert x \rVert - \delta)/\sigma\bigr)$ otherwise.

Attraction forces apply to kNN neighbors in $\Gamma$, while all other pairs experience repulsion. Layout iterations run until a preset number of steps is reached.

# Quantitative measures

## Measuring the global structure

### Persistence diagrams

Compute Vietoris–Rips complexes $VR_t(X)$ and track homology groups $H_k(VR_t(X);\mathbb{F})$ across scales, extracting birth–death pairs $(b_\alpha,d_\alpha)$. Compare diagrams via bottleneck and Wasserstein distances [@persim-docs].

### Betti curves

Plot Betti numbers $\beta_k(t) = \mathrm{rank} H_k(VR_t(X))$ and compare curves using Dynamic Time Warping (DTW) [@salvador-fastdtw], Time Warp Edit Distance (TWED) [@marteau-twed], and Earth Mover's Distance (EMD) [@flamary-pot].

### Global structure preservation

For embedding $Y=f(X)$, measure $d_b(D_k(X), D_k(Y))$ and $d_W(D_k(X),D_k(Y))$ for $k=0,1$, normalized by sample size. For Betti curves, normalize mass before EMD and rescale distances appropriately.

## Measuring the local structure

### Embedding stress

For each edge $e=(x,y) \in \Gamma$, define local stress:

$$
\lambda(e) = 
\left|
  1 - \frac{\lVert x-y\rVert_2}{\lVert f(x)-f(y)\rVert_2}
\right|
$$

and total stress:

$$
\sigma(f,\Gamma) = \frac{\sqrt{\mathrm{Var}\Lambda}}{\mathbb{E}\Lambda},
$$

where $\Lambda = (\lambda(e))_{e\in\text{edges}(\Gamma)}$.

### Neighborhood preservation

Compute neighbor sets $N_X(x)$ and $N_Y(f(x))$. Define preservation sequence $N(X,f)=\bigl(\tfrac{|N_X(x)\cap N_Y(f(x))|}{|N_X(x)|}\bigr)_{x\in X}$ and index:

$$
\nu(X,f) = \left(\mathbb{E} N(X,f),\sqrt{\mathrm{Var}N(X,f)}\right).
$$

## Measuring context loss

### Linear SVM classifier accuracy

Train/test linear SVM on $X$ and $Y$, yielding accuracies $\alpha_X,\alpha_Y$. Define context loss:

$$
\kappa_{\mathrm{SVM}} = \log\min\left(\tfrac{\alpha_X}{\alpha_Y},\tfrac{\alpha_Y}{\alpha_X}\right).
$$

### $k$NN classifier accuracy

Compare $k$NN classifier accuracies $\alpha_X,\alpha_Y$ and define:

$$
\kappa_{k\mathrm{NN}} = \log\frac{\alpha_Y}{\alpha_X}.
$$

# General workflow

1. **Data Preprocessing:** External; users may normalize or transform data via scikit-learn.  
2. **Embedding:** Instantiate `DiRe` and call `fit_transform`.  
3. **Visualization:** Call `visualize` on the returned object.  
4. **Metrics:** Access quantitative measures for analysis and hyperparameter tuning.

# Code availaility
The DiRe - JAX workflow is publicly available on GitHub at [https://github.com/sashakolpakov/dire-jax](https://github.com/sashakolpakov/dire-jax).

# Benchmarks

## Dataset: Blobs

+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ![Blobs DiRe–JAX embedding](pics/embeddings/blobs-dire-jax.png){height="80pt"}   | ![Blobs tSNE embedding](pics/embeddings/blobs-tsne.png){height="80pt"}             |
+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ![Blobs cuML UMAP embedding](pics/embeddings/blobs-cuml-umap.png){height="80pt"} | ![Blobs UMAP embedding](pics/embeddings/blobs-umap.png){height="80pt"}             |
+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

## Dataset: MNIST Digits

+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ![MNIST DiRe–JAX embedding](pics/embeddings/mnist-dire-jax.png){height="80pt"}   | ![MNIST tSNE embedding](pics/embeddings/mnist-tsne.png){height="80pt"}             |
+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ![MNIST cuML UMAP embedding](pics/embeddings/mnist-cuml-umap.png){height="80pt"} | ![MNIST UMAP embedding](pics/embeddings/mnist-umap.png){height="80pt"}             |
+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

## Dataset: Disk Uniform

+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ![Disk DiRe–JAX embedding](pics/embeddings/disk-dire-jax.png){height="80pt"}     | ![Disk tSNE embedding](pics/embeddings/disk-tsne.png){height="80pt"}               |
+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ![Disk cuML UMAP embedding](pics/embeddings/disk-cuml-umap.png){height="80pt"}   | ![Disk UMAP embedding](pics/embeddings/disk-umap.png){height="80pt"}               |
+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

## Dataset: Two Half–Moons

+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ![Moons DiRe–JAX embedding](pics/embeddings/moons-dire-jax.png){height="80pt"}   | ![Moons tSNE embedding](pics/embeddings/moons-tsne.png){height="80pt"}             |
+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ![Moons cuML UMAP embedding](pics/embeddings/moons-cuml-umap.png){height="80pt"} | ![Moons UMAP embedding](pics/embeddings/moons-umap.png){height="80pt"}             |
+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

## Dataset: Levine 13

+-----------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ![Levine 13 DiRe–JAX embedding](pics/embeddings/levine13-dire-jax.png){height="80pt"}   | ![Levine 13 tSNE embedding](pics/embeddings/levine13-tsne.png){height="80pt"}      |
+-----------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ![Levine 13 cuML UMAP embedding](pics/embeddings/levine13-cuml-umap.png){height="80pt"} | ![Levine 13 UMAP embedding](pics/embeddings/levine13-umap.png){height="80pt"}      |
+-----------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

## Dataset: Levine 32

+-----------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ![Levine 32 DiRe–JAX embedding](pics/embeddings/levine32-dire-jax.png){height="80pt"}   | ![Levine 32 tSNE embedding](pics/embeddings/levine32-tsne.png){height="80pt"}      |
+-----------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| ![Levine 32 cuML UMAP embedding](pics/embeddings/levine32-cuml-umap.png){height="80pt"} | ![Levine 32 UMAP embedding](pics/embeddings/levine32-umap.png){height="80pt"}      |
+-----------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+


# Future work

1. Add more quantitative measures and hyperparameter optimization features.  
2. Improve performance of persistent homology computations beyond dimensions 0 and 1.  
3. Explore alternative distance metrics for kNN graph construction and embedding space.

# Acknowledgements

This work is supported by the Google Cloud Research Award number GCP19980904.

# References
