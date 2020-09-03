# VAE-Based Unsupervised Representation Learning for Image-Based Cell Profiling Field
***
## Summary
In the context of image-based cell profiling, measurements of relevant properties of cells are extracted from microscopy images and used as a source of quantitative information to characterize the cell states. Although the recent advancements in deep learning based on convolutional neural networks (CNNs) that automatically learn features from image data, the gold standards in the field still rely on human-designed features. Here, we used a Variational Autoencoder (VAE) maximizing mutual information framework to learn high-quality representations of cells, automatically learnt from the single cell images in one cohesive step. We proposed different quantitive metrics to evaluate the quality of latent representations produced by different techniques and showed that, depending on the dataset characteristics, VAEs can lead to a data embedding of a quality comparable to the traditional methods, while providing several benefits thanks to the learnt from data nature of the process. Moreover, we discussed possible extensions of our framework that would allow to integrate feedbacks from experts throughout visual interaction and to obtain more interpretable features, the lack of the latter being one of the biggest obstacles of the use of artificial intelligence in this field. This improved framework could provide researchers with a new powerful tool to analyze cellular phenotypes effectively and to gain insight into cellular structure variations that characterizes
different populations of cells, highly valuable for many downstream analyses.

![image Pip 1](Code/fig/abstractFigure.png)

---
## Git `Code` Directory Structure
```bash
.
├── human_guidance
│   ├── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
│   ├── feedback_helpers.py
│   └── VAE_feedback_framework.py
├── InfoMAX_VAE_framework.py
├── models
│   ├── infoMAX_VAE.py
│   ├── networks.py
│   ├── nn_modules.py
│   └── train_net.py
├── optimization
│   ├── heatmap_optimization.py
│   ├── hyper_optimization_bs_fixed.py
│   ├── hyper_optimization.py
│   ├── score_optimization.py
│   └── UMAP_TSNE_optimization.py
├── quantitative_metrics
│   ├── backbone_metric.py
│   ├── classifier_metric.py
│   ├── local_quality.py
│   ├── MINE_metric.py
│   ├── performance_metrics.py
│   └── unsupervised_metric.py
├── README.md
├── util
│   ├── data_processing.py
│   ├── file_size_distribution.py
│   ├── helpers.py
│   ├── Process_Chaffer_Dataset.py
│   ├── Process_DataSet_1.py
│   └── Process_Horvath_Dataset.py
└── VAE_framework.py
```
---
## Project Outline

The aim is to investigate and implement CNNs-based deep learning strategies to extract features and embed single-cell images in a 3D latent space in a learnt from data manner, and compare performances with gold standard (tSNE or UMAP projection on human-designed features).
The main challenges are to :
  1. Design a CNN-based neural network that can generate a useful leart representation.
  2. Find or build synthetic datasets with full knowledge to validate our approach.
  3. Implement quantitative metrics to assess the quality of the learnt representation.
  4. Develop a framework that would facilitate the incorporation of human guidance and feature interpretability.

### Datasets
* **BBBC Dataset** : Synthetic Dataset provided by the Broad Bioimage Benchmark Collection ([BBBC031v1](https://bbbc.broadinstitute.org/BBBC031)). 8k Cells divided in 6 continuous biological processes. Latent space is expected to show trajectories.
* **Horvath Dataset** : Subsample of a synthetic dataset first described by Perter Horvath in a [publication](https://linkinghub.elsevier.com/retrieve/pii/S2405471217302272). 8k that are divided in 6 distinct classes. The latent space is expected to present series of discrete states rather than a continuous underlying manifold.
* **Chaffer Dataset** : Unpublished work from Christine Chaffer and John Lock. Real-world fluorescent microscopy single-cells dataset that are expected to be divided in 8 prospective clusters based on underlying biological knowledge.

### VAE Variants
We considered 3 different VAE variants in this project, and compared them to gold standard.
* **Vanilla VAE** (`networks.py`) : Traditional VAE as proposed in its original [publication](https://arxiv.org/abs/1312.6114).
* **SC-VAE** (`networks.py`) : Framework as in [Zheng and al.](https://arxiv.org/abs/1802.06677), in which downsampling (or upsampling) skip connection is added between each layer, both in encoder and decoder.
* **InfoMax VAE** (`infoMAX_VAE.py`) : Framework as in [Rezaabad and al.](https://arxiv.org/abs/1912.13361), that jointly optimize a mutual information estimator. We modified the variational lower bound on mutual information, to use an interpolated bound instead of NWJ bound, as proposed by [Ben Poole](https://arxiv.org/abs/1905.06922).

### Performance metrics
We implemented different quantitative metrics to evaluate the quality of the learnt representation. Defining what is a good and useful latent representation is complex, case-driven and relies on various proxies and on the prior knowledge.
A single metric assesses a single desirable aspect of the projection, and it's a combination of different metrics that can characterize all together a *good* latent representation.

All the metrics can be computed on the projection of any methods, in the file `performance_metrics.py`.

* **Unsupervised metrics** (`unsupervised_metric.py`) : Trustworthiness, Continuity and Local Continuity Meta-Criterion are computed. That are all rank-based criterion that evaluate different aspects of the neighborhood preservation. A local quality score is also computed for each sampled, to plot the projection color-coded by the local neighborhood preservation.
* **Mutual Information** (`MINE_metric.py`) : Mutual information between the inputs and the latent codes was used as proxy for meaningful latent variables. Thanks to the Mutual Information Neural Estimator ([MINE](https://arxiv.org/abs/1801.04062)) framework, we could estimate this mutual information by the mean of an other neural network.
* **Classifier accuracy** (`classifier_metric.py`) : If dataset is expecte to be divided in classes and the class identities are available as ground truth, supervised classification accuracy can gauge the ability of the VAE to structure the latent space in order to discriminate well between the classes.
* **Backbone metric** (`backbone_metric.py`) : If the latent space is expected to reveal an underlying manifold or trajectory, we can use any proxy related to a smooth evolution of features (time, concentration of treatments...) as a ground truth for continuity. The projection of this expected continuum depict a backbone, that can help to visualize the structure of the latent space, and assess if the smooth evolution assumed by the ground truth is preserved.

---

## Installation

To be able to run properly the files, a default Conda environment with python=3.6 is sufficient, with the addition of the following packages and their versions :
* pytorch                   1.4.0
* torchvision               0.2.2
* torchsummary              1.5.1
* scikit-image              0.16.2
* scikit-learn              0.22.2
* scipy                     1.2.1
