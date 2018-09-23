import numpy as np
from collections import defaultdict, Counter
import random
import json

from multiprocessing import Pool
from time import time
from itertools import product

def subgraph(node, connections, k=300, depth=2, mode='random'):
    '''
    random walks,
    paths and endpoints
    '''
    initial = node
    graph = defaultdict(list) 
    graph[node] = [[]]
    if mode == 'random':
        for i in range(k):
            path = []
            node = initial
            for d in range(depth):
                if len(connections[node]) == 0:
                    break
                if d > 0:
                    # print len(connections[node])
                    step = random.choice(connections[node] + ['STOP'])
                    if step == 'STOP':
                        # print 'here'
                        break
                    else:
                        path.append(step)
                else:
                    step = random.choice(connections[node])
                    path.append(step)
                node = path[-1][1]

            if len(path) != 0:
                endpoint = path[-1][1]
                graph[endpoint].append(path)

    else:
        # BFS
        nodes = set([node])
        for d in range(depth):
            new_nodes = set()
            for n in nodes:
                until_n = graph[n]
                if len(until_n) == 0:
                    until_n = [[]]
                for out_edge in connections[n]:
                    new_nodes.add(out_edge[1])
                    for path in until_n:
                        if len(path) < depth:
                            graph[out_edge[1]].append(path + [out_edge[0]])
            nodes = new_nodes

    return graph


def combine_vocab(rel2id_path, ent2id_path, rel_emb, ent_emb, symbol2id_path, symbol2vec_path):
    symbol_id = {}
    rel2id = json.load(open(rel2id_path))
    ent2id = json.load(open(ent2id_path))

    # the relation id will remain unchanged
    for key, idx in rel2id.items():
        symbol_id[key] = idx
    num_rel = len(rel2id.keys())
    for key, idx in ent2id.items():
        symbol_id[key] = idx + num_rel

    num_symbols = len(symbol_id.keys())

    symbol_id['PAD'] = num_symbols

    # print symbol_id['PAD'] # PAD = 69557

    rel_embed = np.loadtxt(rel_emb)
    ent_embed = np.loadtxt(ent_emb)

    symbol_embed = np.concatenate([rel_embed, ent_embed, np.zeros((1, rel_embed.shape[1]))])

    assert symbol_embed.shape[0] == len(symbol_id.keys()) == (num_symbols + 1)
    np.savetxt(symbol2vec_path, symbol_embed)
    json.dump(symbol_id, open(symbol2id_path, 'w'))

class Graph(object):
    """methods to process KB"""
    def __init__(self, path):
        super(Graph, self).__init__()
        self.triples = [] # the graph for path finding
        self.dataset = path
        with open(path + '/path_graph') as f:
            for line in f:
                e1 = line.rstrip().split('\t')[0]
                rel = line.rstrip().split('\t')[1]
                e2 = line.rstrip().split('\t')[2]
                # if rel != relation:
                    # self.triples.append([e1, rel, e2])
                self.triples.append([e1, rel, e2])

        self.connections = defaultdict(list)
        self._connections = defaultdict(list) # inverse connections

        for triple in self.triples:
            e1, rel, e2 = triple
            self.connections[e1].append((rel, e2))
            self._connections[e2].append((rel, e1))

        self.symbol2id = json.load(open(path + '/symbol2ids'))

    def uni_search(self, node_pair, depth=3):
        paths = set()
        start = node_pair[0]
        end = node_pair[1]
        curr_layer = set([start])
        path_tracker = defaultdict(set)
        for d in range(depth):
            new_layer = set()
            new_tracker = defaultdict(set)
            for node in curr_layer:
                # Stop when node has a very large fan-out
                # print 'fan-out: ', len(self.connections[node])
                if len(self.connections[node]) > 1000:
                    continue

                until_node = path_tracker[node]
                if len(until_node) == 0:
                    until_node = set([''])
                for out in self.connections[node]:
                    if out[1] == end:
                        for path in until_node:
                            paths.add(path + ' - ' + out[0] + ' - ' + out[1])
                    else:
                        new_layer.add(out[1])
                        for path in until_node:
                            new_tracker[out[1]].add(path + ' - ' + out[0] + ' - ' + out[1])
            curr_layer = new_layer
            path_tracker = new_tracker

        paths = [node_pair[0] + ' - ' + item[1:] for item in paths]
        return paths  

    def path_clean(self, path):
        symbols = path.split(' - ')
        # print 'Uncleaned path: ', symbols   
        entities = []
        for idx, item in enumerate(symbols):
            if idx%2 == 0:
                entities.append(item)
        entity_stats = Counter(entities).items()
        duplicate_ents = [item for item in entity_stats if item[1]!=1]
        duplicate_ents.sort(key = lambda x:x[1], reverse=True)
        for item in duplicate_ents:
            ent = item[0]
            ent_idx = [i for i, x in enumerate(symbols) if x == ent]
            if len(ent_idx)!=0:
                min_idx = min(ent_idx)
                max_idx = max(ent_idx)
                if min_idx!=max_idx:
                    symbols = symbols[:min_idx] + symbols[max_idx:]

        # print 'cleaned path: ', symbols
        return ' - '.join(symbols)

    def pair_feature(self, pair, k=300, depth=3, mode='random'):
        '''
        some graph patterns around two entity nodes
        how many random walks: k
        '''
        # # generate subgraph for left/right node
        # p = Pool(processes=2)
        # graphs = p.starmap(subgraph, [(pair[0], self.connections, k, depth), (pair[1], self._connections, k, depth)])
        # p.close()
        graphs = []
        graphs.append(subgraph(pair[0], self.connections, k, depth, mode=mode))
        graphs.append(subgraph(pair[1], self._connections, k, depth, mode=mode))

        # combine two subgraphs
        left = graphs[0]
        right = graphs[1]

        intermediate = set(left.keys()).intersection(set(right.keys()))
        paths = []
        if len(list(intermediate)) == 0:
            return paths
        for node in intermediate:
            l = left[node]
            r = right[node]
            combinations = product(l, r)
            for _ in combinations:
                # paths.append(_[0] + _[1][::-1]
                sub_1 = _[0]
                sub_2 = _[1]

                if len(sub_1) != 0:
                    sub_path_1 = []
                    for rel, ent in sub_1:
                        sub_path_1.append(rel)
                        sub_path_1.append(ent)
                    assert sub_path_1[-1] == node
                    sub_path_1 = pair[0] + ' - ' + ' - '.join(sub_path_1)
                else:
                    sub_path_1 = pair[0]

                if len(sub_2) != 0:
                    sub_path_2 = []
                    for rel, ent in sub_2:
                        sub_path_2.append(rel)
                        sub_path_2.append(ent)
                    sub_path_2 = sub_path_2[::-1]
                    assert sub_path_2[0] == node
                    sub_path_2 = ' - '.join(sub_path_2[1:]) + ' - ' + pair[1]
                else:
                    sub_path_2 = ''

                if sub_path_2 == '':
                    path = sub_path_1
                else: 
                    path = sub_path_1 + ' - ' + sub_path_2
                paths.append(path)

        # path cleaning
        paths = list(set(paths))
        cleaned_paths = []
        for path in paths:
            cleaned_paths.append(self.path_clean(path))
                                
        return list(set(cleaned_paths))

    def encode_path(self, path, seq_len=13):
        encoded = []
        path = path.split(' - ')
        for symbol in path:
            encoded.append(self.symbol2id[symbol])
        encoded = np.pad(encoded, (0, seq_len-len(encoded)), 'constant', constant_values=(0, self.symbol2id['PAD']))
        return encoded  


    def train_generate(self, few=5, batch_size=50, num_neg=1):
        '''
        data generator for training
        '''
        dataset = self.dataset
        print 'LOAD TRAINING DATA'
        train_tasks = json.load(open(dataset + '/train_tasks.json'))

        print 'BUILD CANDAIDATES FOR EVERY RELATION'
        rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
        task_pool = list(train_tasks.keys())

        num_tasks = len(task_pool)
        rel_idx = 0

        while True:
            # sample a task (relation to do reasoning)
            if rel_idx % num_tasks == 0:
                random.shuffle(task_pool)
            query = task_pool[rel_idx % num_tasks]
            rel_idx += 1
            tail_candidates = rel2candidates[query]
            # if len(train_tasks[query]) > few + batch_size:
            #     train_and_test = random.sample(train_tasks[query], few+batch_size)
            # else:
            #     train_and_test = train_tasks[query]
            #     random.shuffle(train_and_test)

            train_and_test = train_tasks[query]

            support = train_and_test[:few]
            test_data = train_and_test[few:]

            support_paths = []
            for triple in support:
                e_h = triple[0]
                e_t = triple[2]
                paths = self.pair_feature((e_h, e_t))
                for path in paths:
                    support_paths.append(self.encode_path(path))

            if len(support_paths) == 0:
                print 'NO PATH FOUND, TRY AGAIN'
                continue

            test_pos_paths = []
            test_neg_paths = []
            for triple in test_data:
                e_h = triple[0]
                e_t = triple[2]
                paths = self.pair_feature((e_h, e_t))
                paths_encoded = []
                for path in paths:
                    paths_encoded.append(self.encode_path(path))
                test_pos_paths.append(paths_encoded)

                while True:
                    noise = random.choice(tail_candidates)
                    if noise != e_t:
                        break
                paths = self.pair_feature((e_h, noise))
                paths_encoded = []
                for path in paths:
                    paths_encoded.append(self.encode_path(path))
                test_neg_paths.append(paths_encoded)

            assert len(test_pos_paths) == len(test_neg_paths)    
            yield support_paths, test_pos_paths, test_neg_paths



if __name__ == '__main__':
    combine_vocab('NELL/relation2ids_fix', 'NELL/ent2ids_fix', 'NELL/relation2vec_fix.bern', 'NELL/entity2vec_fix.bern', 'NELL/symbol2ids_fix', 'NELL/symbol2vec_fix.txt')
