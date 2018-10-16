# One-Shot-Knowledge-Graph-Reasoning

PyTorch implementation of the One-Shot relational learning model described in our EMNLP 2018 paper [One-Shot Relational Learning for Knowledge Graphs](https://arxiv.org/abs/1808.09040). In this work, we attempt to automatically infer new facts about a particular relation given only one training example. For instance, given the fact the "the Arlanda Airport is located in city Stochholm", the algorithm proposed in this papers tries to automatically infer that "the Haneda Airport is located in Tokyo" by utilizing the knowledge graph information about the involved entities (i.e. the Arlanda Airport, Stochholm, the Haneda Airport and Tokyo).

## Method illustration

<p align="center"><img width="85%" src="imgs/all.png" /></p>

The main idea of this model is a matching network that encodes the one-hop neighbors of the involved entities, as defined in ``matcher.py``.

## Steps to run the experiments

### Requirements
* ``Python 3.6.5 ``
* ``PyTorch 0.4.1``
* ``tensorboardX``
* ``tqdm``

### Datasets
Download datasets [Wiki-One](http://nlp.cs.ucsb.edu/data/wiki.tar.gz) or [NELL-One](http://nlp.cs.ucsb.edu/data/nell.tar.gz)

### Pre-trained embeddings


### Training
* With random initialized embeddings: ``CUDA_VISIBLE_DEVICES=0 python trainer.py --max_neighbor 50  --fine_tune``
* With pretrained embeddings: ``CUDA_VISIBLE_DEVICES=0 python trainer.py --max_neighbor 50  --fine_tune --embed_model ComplEx``

### Visualization
``tensorboard --logdir logs``

### Reference
```
@article{xiong2018one,
  title={One-Shot Relational Learning for Knowledge Graphs},
  author={Xiong, Wenhan and Yu, Mo and Chang, Shiyu and Guo, Xiaoxiao and Wang, William Yang},
  journal={arXiv preprint arXiv:1808.09040},
  year={2018}
}
```
