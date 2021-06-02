# A Deep Latent Space Model for Graph Representation Learning

This is the source code of our work "A Deep Latent Space Model for Graph Representation Learning".

## Abstract

Graph representation learning is a fundamental problem for modeling relational data and benefits a number of downstream applications. Traditional Bayesian-based graph models and recent deep learning based GNN either suffer from impracticability or lack interpretability, thus combined models for undirected graphs have been proposed to overcome the weaknesses. As a large portion of real-world graphs are directed graphs (of which undirected graphs are special cases), in this paper, we propose a Deep Latent Space Model (DLSM) for directed graphs to incorporate the traditional latent variable based generative model into deep learning frameworks. Our proposed model consists of a graph convolutional network (GCN) encoder and a latent space model (LSM) based decoder, which are connected by a novel hierarchical variational auto-encoder architecture. By specifically modeling the degree heterogeneity using node random factors, our model possesses better interpretability in both community structure and degree heterogeneity. For fast inference, the stochastic gradient variational Bayes (SGVB) is adopted using a non-iterative recognition model, which is much more scalable than traditional MCMC-based methods. The experiments on real-world datasets show that the proposed model achieves the state-of-the-art performances on both link prediction and community detection tasks while learning interpretable node embeddings.

## Requirements

python 3.7.6

tensorflow 2.2.0

## Examples

python train.py --model dlsm --dataset email --epochs 2000 --encoder 32_64_128 --decoder 50_100 --latent_dim 50 --directed 1 --features 0 --learning_rate 0.01

python train.py --model dlsm_ip --dataset kohonen --epochs 2000 --encoder 32_64_128 --decoder 50_100 --directed 1 --learning_rate 0.01
