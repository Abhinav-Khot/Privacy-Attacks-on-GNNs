## Testing Model Inversion Attacks agaisnt DPAR (Decoupled GNN with Differentially Private Approximate Personalized PageRank)

This repository outlines the testing of model inversion privacy attacks on the DPAR architecture under different settings. This attack tries to reconstruct the training graph based on model weights and a small amount of feature and label information.

The repository has 3 branches and each of them explore different experiments which try to deter the attack such as:
- Effect of sampling the graph before training
- Effect of dropping edges
- Effect of rewiring edges


The base code for the privacy attack is adapted from the following paper:
```
@inproceedings{ijcai2021-516,
  title     = {GraphMI: Extracting Private Graph Data from Graph Neural Networks},
  author    = {Zhang, Zaixi and Liu, Qi and Huang, Zhenya and Wang, Hao and Lu, Chengqiang and Liu, Chuanren and Chen, Enhong},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  pages     = {3749--3755},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/516},
  url       = {https://doi.org/10.24963/ijcai.2021/516},
}
```
