# A Deep Latent Space Model for Interpretable Representation Learning on Directed Graphs

This is the source code of our work "A Deep Latent Space Model for Interpretable Representation Learning on Directed Graphs" for medium-sized graph datasets. [Here](https://github.com/upperr/DLSM-OGB) is another implementation of our work for the large-scale OGB datasets, which employs GraphSAGE as the encoder within our framework.

## Abstract

Graph representation learning is a fundamental research problem for modeling relational data and benefits a number of downstream applications. Traditional Bayesian-based random graph models, such as the stochastic blockmodels (SBMs) and latent space models (LSMs), have proved effective to learn interpretable representations. To take the advantages of both the random graph models and deep learning-based methods such as graph neural networks (GNNs), some research proposes deep generative methods by combining the SBMs and GNNs. However, these combined methods have not fully considered the statistical properties of graphs which limits the model interpretability and applicability on directed graphs. To address these limitations in existing research, in this paper, we propose a Deep Latent Space Model (DLSM) for interpretable representation learning on directed graphs, by combining the LSMs and GNNs via a novel "lattice VAE" architecture. The proposed model generates multiple latent variables as node representations to adapt to the structure of directed graphs and improve model interpretability. Extensive experiments on representative real-world datasets demonstrate that our model achieves the state-of-the-art performances on link prediction and community detection with good interpretability.

## Requirements

- python 3.7.6

- tensorflow 2.2.0

## Examples

### Link Prediction

```
python train.py --model dlsm --dataset political --use_kl_warmup 1
```
```
python train.py --model dlsm_d --dataset wiki --use_kl_warmup 0
```

### Community Detection

```
python train.py --model dlsm --dataset political --link_prediction 0 --community_detection 1 --use_kl_warmup 1
```
