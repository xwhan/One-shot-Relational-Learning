'''
prepare the triples for few-shot training
'''
import numpy as np
from collections import defaultdict
import json

from time import time

def build_vocab(dataset):
    rels = set()
    ents = set()

    with open(dataset + '/path_graph') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            rel = line.split('\t')[1]
            e1 = line.split('\t')[0]
            e2 = line.split('\t')[2]
            rels.add(rel)
            rels.add(rel + '_inv')
            ents.add(e1)
            ents.add(e2)

    relationid = {}
    for idx, item in enumerate(list(rels)):
        relationid[item] = idx
    entid = {}
    for idx, item in enumerate(list(ents)):
        entid[item] = idx

    json.dump(relationid, open(dataset + '/relation2ids', 'w'))
    json.dump(entid, open(dataset + '/ent2ids', 'w'))            

def candidate_triples(dataset):
    '''
    build candiate tail entities for every relation
    '''
    # calculate node degrees
    with open(dataset + '/path_graph') as f:
        for line in f:
            line = line.rstrip()
            e1 = line.split('\t')[0]
            e2 = line.split('\t')[2]

    ent2ids = json.load(open(dataset+'/ent2ids'))

    all_entities = ent2ids.keys()

    type2ents = defaultdict(set)
    for ent in all_entities:
        try:
            type_ = ent.split(':')[1]
            type2ents[type_].add(ent)
        except Exception as e:
            continue

    # train_tasks = json.load(open(dataset + '/train_tasks.json'))
    train_tasks = json.load(open(dataset + '/known_rels.json'))

    dev_tasks = json.load(open(dataset + '/dev_tasks.json'))
    test_tasks = json.load(open(dataset + '/test_tasks.json'))

    all_reason_relations = train_tasks.keys() + dev_tasks.keys() + test_tasks.keys()

    all_reason_relation_triples = train_tasks.values() + dev_tasks.values() + test_tasks.values()

    assert len(all_reason_relations) == len(all_reason_relation_triples)

    rel2candidates = {}
    for rel, triples in zip(all_reason_relations, all_reason_relation_triples):

        possible_types = set()
        for example in triples:
            try:
                type_ = example[2].split(':')[1] # type of tail entity
                possible_types.add(type_)
            except Exception as e:
                print example

        candidates = []
        for type_ in possible_types:
            candidates += list(type2ents[type_])
        
        rel2candidates[rel] = list(set(candidates))

    json.dump(rel2candidates, open(dataset + '/rel2candidates_all.json', 'w'))

def wiki_candidate(dataset):
    ent2id = json.load(open(dataset + '/ent2ids'))
    type2ents = defaultdict(list)
    ent2type = {}
    with open(dataset + '/instance_of') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            type_ = line.split()[2]
            ent = line.split()[0]
            if ent in ent2id:
                type2ents[type_].append(ent)
                ent2type[ent] = type_


    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    dev_tasks = json.load(open(dataset + '/dev_tasks.json'))
    test_tasks = json.load(open(dataset + '/test_tasks.json'))

    all_reason_relations = train_tasks.keys() + dev_tasks.keys() + test_tasks.keys()

    all_reason_relation_triples = train_tasks.values() + dev_tasks.values() + test_tasks.values()

    print 'How many few-shot relations', len(all_reason_relations)

    rel2candidates = {}
    for rel, triples in zip(all_reason_relations, all_reason_relation_triples):

        possible_types = []
        for example in triples:
            possible_types.append(ent2type[example[2]])

        possible_types = set(possible_types)

        candidates = []
        for _ in possible_types:
            candidates += type2ents[_]
        
        candidates = list(set(candidates))
        if len(candidates) > 5000:
            candidates = candidates[:5000]
        rel2candidates[rel] = candidates

    json.dump(rel2candidates, open(dataset + '/rel2candidates.json', 'w'))

    dev_tasks_ = {}
    test_tasks_ = {}

    for key, triples in dev_tasks.items():
        # print len(rel2candidates[key])
        if len(rel2candidates[key]) < 20:
            continue
        dev_tasks_[key] = triples

    for key, triples in test_tasks.items():
        # print len(rel2candidates[key])
        if len(rel2candidates[key]) < 20:
            continue
        test_tasks_[key] = triples

    json.dump(dev_tasks_, open(dataset + '/dev_tasks.json', 'w'))
    json.dump(test_tasks_, open(dataset + '/test_tasks.json', 'w'))

def combine_vocab(rel2id_path, ent2id_path, rel_emb, ent_emb, symbol2id_path, symbol2vec_path):
    symbol_id = {}

    print 'LOADING SYMBOL2ID'
    rel2id = json.load(open(rel2id_path))
    ent2id = json.load(open(ent2id_path))

    # print set(rel2id.keys()) & set(ent2id.keys()) # '' and 'OOV'
    print 'LOADING EMBEDDINGS'
    rel_embed = np.loadtxt(rel_emb, dtype=np.float32)
    ent_embed = np.loadtxt(ent_emb, dtype=np.float32)

    assert rel_embed.shape[0] == len(rel2id.keys())
    assert ent_embed.shape[0] == len(ent2id.keys())   

    i = 0
    embeddings = []
    for key in rel2id.keys():
        if key not in ['','OOV']:
            symbol_id[key] = i
            i += 1
            embeddings.append(list(rel_embed[rel2id[key],:]))

    for key in ent2id.keys():
        if key not in ['', 'OOV']:
            symbol_id[key] = i
            i += 1
            embeddings.append(list(ent_embed[ent2id[key],:]))

    symbol_id['PAD'] = i
    embeddings.append(list(np.zeros((rel_embed.shape[1],))))
    embeddings = np.array(embeddings)

    assert embeddings.shape[0] == len(symbol_id.keys())
    np.savetxt(symbol2vec_path, embeddings)
    json.dump(symbol_id, open(symbol2id_path, 'w'))


def freq_rel_triples(dataset):
    known_rels = defaultdict(list)
    with open(dataset + '/path_graph') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            e1,rel,e2 = line.split()
            known_rels[rel].append([e1,rel,e2])

    train_tasks = json.load(open(dataset + '/train_tasks.json'))

    for key, triples in train_tasks.items():
        known_rels[key] = triples

    json.dump(known_rels, open(dataset + '/known_rels.json', 'w'))


def for_filtering(dataset, save=False):
    e1rel_e2 = defaultdict(list)
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    dev_tasks = json.load(open(dataset + '/dev_tasks.json'))
    test_tasks = json.load(open(dataset + '/test_tasks.json'))
    few_triples = []
    for _ in (train_tasks.values() + dev_tasks.values() + test_tasks.values()):
        few_triples += _
    for triple in few_triples:
        e1,rel,e2 = triple
        e1rel_e2[e1+rel].append(e2)
    if save:
        json.dump(e1rel_e2, open(dataset + '/e1rel_e2.json', 'w'))


if __name__ == '__main__':
    start = time()

    # candidate_triples('NELL')
    # build_vocab('Wiki')
    combine_vocab('Wiki/relation2ids', 'Wiki/ent2ids', 'Wiki/relation2vec.vec', 'Wiki/entity2vec.vec', 'Wiki/symbol2ids', 'Wiki/symbol2vec.vec')
    # wiki_candidate('Wikidata')

    # freq_rel_triples('NELL')

    # candidate_triples('NELL')

    # combine_vocab('NELL/relation2ids', 'NELL/ent2ids', 'NELL/relation2vec.vec', 'NELL/entity2vec.vec', 'NELL/symbol2ids', 'NELL/symbol2vec.vec')

    print 'Time clipse: ', time() - start

