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


### Experiments
1. train a simple net (3 blocks like resnet), on cifar10 with accuracy of 86%. 
after this, I trained the model only on 2 classes from the dataset (cat, dog - 5000 images per class), with accuracy of ~80%.

2. Used SiameseNet for Discriminator, takes two representations of a batch (tensors) and return similarity score. SiameseNet contains layers of Max Polling.
   According to the DCGAN paper, it is not recommended using these layers while training a GAN, and replace the down/up sampling with stride conv layers.
   # todo - change the architecture


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

#### Siamese Neural Networks for One-shot Image Recognition (2015)

- Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton.
  Google Research, Brain Team.
- Accepted to ICML.
- [paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [code](https://github.com/fangpin/siamese-pytorch)

Siamese network uses a supervised training approach to learn generic input features then, based on the training data, it makes predictions about unknown class distributions.

1. Siamese network takes two different inputs passed through two similar subnetworks with the same architecture, parameters, and weights.
2. The two subnetworks are a mirror image of each other, just like the Siamese twins. Hence, any change to any subnetworks architecture, parameter, or weights is also applied to the other subnetwork.
3. The Siamese network's objective is to classify if the two inputs are the same or different using the Similarity score. 
  The Similarity score can be calculated using Binary cross-entropy, Contrastive function, or Triplet loss, which are techniques for the general distance metric learning approach.
  

<p align="center">
<img src="images/siam_arc.jpeg" alt="Siamese Architecture" width="70%"/>
</p>


#### Let’s Agree to Agree: Neural Networks Share Classification Order on Real Datasets (2020)

- Guy Hacohen, Leshem Choshen, Daphna Weinshall 
  School of Computer Science and Engineering, The Hebrew University of Jerusalem.
- Accepted to ICML.
- [paper](https://arxiv.org/pdf/1905.10854.pdf)

Deep Neural Networks learn the examples in both the training and test sets in a similar order.
models of different architectures start by learning the same examples,
after which the more powerful model may continue to learn additional examples.

1. Direct way to compare between different neural models termed TP-agreement (accuracy per image).
2. Models that share the same architecture learn real datasets in the same order
3. Stronger architectures start off by learning the same examples that weaker networks learn, then move on to learning new examples.

