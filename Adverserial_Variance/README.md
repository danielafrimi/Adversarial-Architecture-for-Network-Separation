# Thesis

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Experiments](#experiments)
  - [Getting Started](#getting-started)
    - [Creating The Environment](#creating-the-environment)
      - [CUDA >= 10.2](#cuda--102)
      - [CUDA 9](#cuda-9)
    - [wandb](#wandb)
    - [Clone The Repo](#clone-the-repo)
    - [Running The Experiments](#running-the-experiments)
      - [Prerequisites](#prerequisites)
      - [Example Command-lines](#example-command-lines)
  - [1st iteration](#1st-iteration)
  - [2nd iteration](#2nd-iteration)
  - [3rd iteration](#3rd-iteration)
- [Related Work](#related-work)
  - [Representation Learning](#representation-learning)
    - [SimCLR - A Simple Framework for Contrastive Learning of Visual Representations (Feb 2020)](#simclr---a-simple-framework-for-contrastive-learning-of-visual-representations-feb-2020)
    - [SimCLRv2 - Big Self-Supervised Models are Strong Semi-Supervised Learners (Jun 2020)](#simclrv2---big-self-supervised-models-are-strong-semi-supervised-learners-jun-2020)
    

### Related Papers & Subjects

#### Metric Learninig
Many approaches in machine learning require a measure of distance between data points. However, when using traditional methods it is often difficult to design metrics that are well-suited to the particular data and task of interest.

Distance metric learning (or simply, metric learning) aims at automatically constructing task-specific distance metrics from (weakly) supervised data, in a machine learning manner. The learned distance metric can then be used to perform various tasks (e.g., k-NN classification, clustering, information retrieval).

It means learning a distance in a low dimensional space(non-input space) such that similar images in the input space result in similar representation(low distance) and dissimilar images result in varied representation(high distance).

### Representation Learning

#### SimCLR - A Simple Framework for Contrastive Learning of Visual Representations (Feb 2020)

- Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton.  
  Google Research, Brain Team.
- Accepted to ICML 2020.
- [paper](https://arxiv.org/pdf/2002.05709.pdf)
- [code](https://github.com/google-research/simclr)

This paper presents SimCLR: A simple framework for contrastive learning of visual representations. \
The self-supervised task is to identify that different augmentations of the same image are the same.

<p align="center">
<img src="images/simclr_architecture.png" alt="SimCLR Architecture" width="70%"/>
</p>

Take home messages:

- Composition of data augmentations is important.
- Adding a nonlinear transformation between the representation and the contrastive loss helps.
- Contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning.