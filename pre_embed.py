import numpy as np 
from collections import defaultdict, Counter
import random
import json

from tqdm import tqdm


def transX(dataset):
    rel2id = json.load(open(dataset + '/relation2ids'))
    ent2id = json.load(open(dataset + '/ent2ids'))

    with open('../Fast-TransX/' + dataset +  '_base/entity2id.txt', 'w') as g1:
        num_ents = len(ent2id.keys())
        g1.write(str(num_ents) + '\n')
        for k, v in ent2id.items():
            g1.write(k + '\t' + str(v) + '\n')

    with open('../Fast-TransX/' + dataset +  '_base/relation2id.txt', 'w') as g1:
        num_rels = len(rel2id.keys())
        g1.write(str(num_rels) + '\n')
        for k, v in rel2id.items():
            g1.write(k + '\t' + str(v) + '\n')


    file_name = dataset + '/path_graph'
    train_triples = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            e1 = line.split('\t')[0]
            e2 = line.rstrip().split('\t')[2]
            rel = line.split('\t')[1]
            train_triples.append([e1,rel,e2])
            train_triples.append([e2,rel+'_inv',e1])

    with open('../Fast-TransX/' + dataset +  '_base/train2id.txt', 'w') as g3:
        num_triples = len(train_triples)
        g3.write(str(num_triples) + '\n')
        for triple in train_triples:
            e1, rel, e2 = triple
            g3.write(str(ent2id[e1]) + '\t' + str(ent2id[e2]) + '\t' + str(rel2id[rel]) + '\n')


if __name__ == '__main__':
    transX('Wiki')