# Combing Latent Space Models and Graph Neural Networks for Directed Graph Representation Learning

This is the source code of our work "Combing Latent Space Models and Graph Neural Networks for Directed Graph Representation Learning" for medium-scale graph datasets. [Here](https://github.com/upperr/DLSM-OGB) is another implementation of our work for the large-scale OGB datasets, which employs GraphSAGE as the encoder within our framework.

## Abstract

Graph representation learning is a fundamental research problem for modeling relational data and benefits a number of downstream applications. Traditional Bayesian-based random graph models, such as the latent space models (LSMs), have been proved effective to learn interpretable representations, but suffer from the scalability issue. With the powerful representation learning ability of deep learning-based methods, the deep generative methods for graph-structured data, such as the variational auto-encoders (VAEs), have also emerged. However, most existing deep generative methods only consider undirected graphs, and usually bring about the interpretability issue. To overcome the problems in existing research and take the advantages of both random graph models and graph neural networks (GNNs), in this paper, we propose a deep generative model DLSM (Deep Latent Space Model) for directed graph representation learning by combining the LSMs and GNNs via a hierarchical VAE architecture. To adapt to directed graphs, our model generates multiple highly interpretable latent variables as node representations, which is also scalable for large-scale graphs. The experimental results on the real-world datasets demonstrate that our model achieves superior performances on multiple downstream tasks with good interpretability and scalability.

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

### Graph Generation

```
python train.py --model dlsm --dataset email --link_prediction 0 --graph_generation 1 --use_kl_warmup 1
```
