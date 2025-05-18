# Code for KTCG

![Paper](https://img.shields.io/badge/Paper-Knowledge\-Based%20Systems-blue)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)

This repository contains the official implementation of ​**KTCG**.

## Requirements
The code has been tested running under Python 3.8.0. Required packages:
- numpy == 1.22.4
- pandas == 1.4.3
- scikit-learn == 1.1.1
- scipy == 1.7.0
- networkx == 2.5.1
- tqdm == 4.64.1  
- torch == 1.10.1+cu113
- torch-cluster == 1.5.9+pt110cu113
- torch-scatter == 2.0.9+pt110cu113
- torch-sparse == 0.6.12+pt110cu113
- torch-geometric == 1.7.2 

## Dataset

We provide three processed datasets: Book-Crossing, MovieLens-1M, and Last.FM.

We follow the paper " [Ripplenet: Propagating user preferences on the knowledge
graph for recommender systems](https://github.com/hwwang55/RippleNet)." to process data.


|                       |               | Book-Crossing | MovieLens-1M | Last.FM |
| :-------------------: | :------------ | ----------:   | --------: | ---------: |
| User-Item Interaction | #Users        |      17,860   |    6,036  |      1,872 |
|                       | #Items        |      14,910   |    2,347  |      3,846 |
|                       | #Interactions |     139,746   |  753,772  |     42,346 |
|    Knowledge Graph    | #Entities     |      77,903   |    6,729  |      9,366 |
|                       | #Relations    |          25   |        7  |         60 |
|                       | #Triplets     |   19,793      |   20,195  |     15,518 |



## Reference 
- We partially use the codes of [KGIN](https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network).
