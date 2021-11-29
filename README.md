# A Deep Latent Space Model for Directed Graph Representation Learning

This is the source code of our work "A Deep Latent Space Model for Directed Graph Representation Learning".

## Abstract

Graph representation learning is a fundamental problem for modeling relational data and benefits a number of downstream applications. Traditional Bayesian-based random graph models and recent deep learning based methods are complementary to each other in interpretability and scalability. To take the advantages of both models, some combined methods have been proposed. However, existing models are mainly designed for \textit{undirected graphs}, while a large portion of real-world graphs are directed. The focus of this paper is on the more challenging \textit{directed graphs} where both the existences and directions of edges need to be learned. We propose a Deep Latent Space Model (DLSM) for directed graphs to incorporate the traditional latent space random graph model into deep learning frameworks via a hierarchical variational auto-encoder architecture. To adapt to directed graphs, our model generates multiple highly interpretable latent variables as node representations, and the interpretability of representing node influences is theoretically proved. The experimental results on real-world graphs demonstrate that our proposed model achieves the state-of-the-art performances on link prediction and community detection tasks while generating interpretable node representations of directed graphs.

## Requirements

python 3.7.6

tensorflow 2.2.0

## Examples

python train.py --model dlsm --dataset political --epochs 2000 --encoder 32_64_128 --decoder 50_100 --latent_dim 50 --directed 1 --features 0 --learning_rate 0.01 --use_kl_warmup 1
