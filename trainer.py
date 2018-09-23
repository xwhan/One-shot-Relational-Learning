import json
import logging
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from args import read_options
from data_loader import *
from matcher import *
from tensorboardX import SummaryWriter

class Trainer(object):
    
    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items(): setattr(self, k, v)

        self.meta = not self.no_meta

        if self.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True

        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        if self.test or self.random_embed:
            self.load_symbol2id()
            use_pretrain = False
        else:
            # load pretrained embedding
            self.load_embed()
        self.use_pretrain = use_pretrain

        # if self.embed_model == 'RESCAL':
        #     self.num_ents = len(self.ent2id.keys()) - 1
        #     self.pad_id_ent = self.num_ents
        #     self.num_rels = len(self.rel2id.keys()) - 1
        #     self.pad_id_rel = self.num_rels
        #     self.matcher = RescalMatcher(self.embed_dim, self.num_ents, self.num_rels, use_pretrain=self.use_pretrain, ent_embed=self.ent_embed, rel_matrices=self.rel_matrices,dropout=self.dropout, attn_layers=self.n_attn, n_head=self.n_head, batch_size=self.batch_size, process_steps=self.process_steps, finetune=self.fine_tune, aggregate=self.aggregate)
        # else:
        self.num_symbols = len(self.symbol2id.keys()) - 1 # one for 'PAD'
        self.pad_id = self.num_symbols
        self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols, use_pretrain=self.use_pretrain, embed=self.symbol2vec, dropout=self.dropout, batch_size=self.batch_size, process_steps=self.process_steps, finetune=self.fine_tune, aggregate=self.aggregate)
        self.matcher.cuda()

        self.batch_nums = 0
        if self.test:
            self.writer = None
        else:
            self.writer = SummaryWriter('logs/' + self.prefix)

        self.parameters = filter(lambda p: p.requires_grad, self.matcher.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[200000], gamma=0.5)

        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())

        logging.info('BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json')) 

        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

    def load_symbol2id(self):

        # if self.embed_model == 'RESCAL':
        #     self.rel2id = json.load(open(self.dataset + '/relation2ids'))
        #     self.ent2id = json.load(open(self.dataset + '/ent2ids'))

        #     self.rel2id['PAD'] = len(self.rel2id.keys())
        #     self.ent2id['PAD'] = len(self.ent2id.keys())
        #     self.ent_embed = None
        #     self.rel_matrices = None
        #     return
        
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        i = 0
        for key in rel2id.keys():
            if key not in ['','OOV']:
                symbol_id[key] = i
                i += 1

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def load_embed(self):

        # if self.embed_model == 'RESCAL':
        #     self.rel2id = json.load(open(self.dataset + '/relation2ids'))
        #     self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        #     self.rel2id['PAD'] = len(self.rel2id.keys())
        #     self.ent2id['PAD'] = len(self.ent2id.keys())
        #     self.ent_embed = np.loadtxt(self.dataset + '/entity2vec.' + self.embed_model)
        #     self.rel_matrices = np.loadtxt(self.dataset + '/relation2vec.' + self.embed_model)
        #     self.ent_embed = np.concatenate((self.ent_embed, np.zeros((1,self.embed_dim))),axis=0)
        #     self.rel_matrices = np.concatenate((self.rel_matrices, np.zeros((1, self.embed_dim * self.embed_dim))), axis=0)
        #     return    


        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))

        logging.info('LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            ent_embed = np.loadtxt(self.dataset + '/entity2vec.' + self.embed_model)
            rel_embed = np.loadtxt(self.dataset + '/relation2vec.' + self.embed_model)

            if self.embed_model == 'ComplEx':
                # normalize the complex embeddings
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            assert ent_embed.shape[0] == len(ent2id.keys())
            assert rel_embed.shape[0] == len(rel2id.keys())

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

            self.symbol2id = symbol_id
            self.symbol2vec = embeddings

    def build_connection(self, max_=100):

        # if self.embed_model == 'RESCAL':
        #     self.connections = np.ones((self.num_ents, max_, 2)).astype(int)
        #     self.connections[:,:,0] = self.pad_id_rel
        #     self.connections[:,:,1] = self.pad_id_ent
        #     self.e1_rele2 = defaultdict(list)
        #     self.e1_degrees = defaultdict(int)
        #     with open(self.dataset + '/path_graph') as f:
        #         lines = f.readlines()
        #         for line in tqdm(lines):
        #             e1,rel,e2 = line.rstrip().split()
        #             self.e1_rele2[e1].append((self.rel2id[rel], self.ent2id[e2]))
        #             self.e1_rele2[e2].append((self.rel2id[rel+'_inv'], self.ent2id[e1]))  

        # else:
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1,rel,e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                self.e1_rele2[e2].append((self.symbol2id[rel+'_inv'], self.symbol2id[e1]))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            # degrees.append(len(neighbors)) 
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors) # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]

        # json.dump(degrees, open(self.dataset + '/degrees', 'w'))
        # assert 1==2

        return degrees

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.matcher.state_dict(), path)

    def load(self):
        self.matcher.load_state_dict(torch.load(self.save_path))

    def get_meta(self, left, right):
        left_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0))).cuda()
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
        right_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0))).cuda()
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
        return (left_connections, left_degrees, right_connections, right_degrees)

    def train(self):
        logging.info('START TRAINING...')

        best_hits10 = 0.0

        losses = deque([], self.log_every)
        margins = deque([], self.log_every)

        # if self.embed_model == 'RESCAL':
        #     self.symbol2id = self.ent2id

        for data in train_generate(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id, self.e1rel_e2):

            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data

            # TODO more elegant solution
            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            support = Variable(torch.LongTensor(support)).cuda()
            query = Variable(torch.LongTensor(query)).cuda()
            false = Variable(torch.LongTensor(false)).cuda()

            if self.no_meta:
            # for ablation
                query_scores = self.matcher(query, support)
                false_scores = self.matcher(false, support)
            else:
                query_scores = self.matcher(query, support, query_meta, support_meta)
                false_scores = self.matcher(false, support, false_meta, support_meta)

            margin_ = query_scores - false_scores
            margins.append(margin_.mean().item())
            loss = F.relu(self.margin - margin_).mean()
            self.writer.add_scalar('MARGIN', np.mean(margins), self.batch_nums)

            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(self.parameters, self.grad_clip)
            self.optim.step()

            if self.batch_nums % self.eval_every == 0:
                hits10, hits5, mrr = self.eval(meta=self.meta)
                self.writer.add_scalar('HITS10', hits10, self.batch_nums)
                self.writer.add_scalar('HITS5', hits5, self.batch_nums)
                self.writer.add_scalar('MAP', mrr, self.batch_nums)

                self.save()

                if hits10 > best_hits10:
                    self.save(self.save_path + '_bestHits10')
                    best_hits10 = hits10

                # if self.batch_nums % (4 * self.eval_every) == 0:
                #     hits10_, hits5_, mrr_ = self.eval(meta=self.meta, mode='test')
                #     self.writer.add_scalar('HITS10-test', hits10_, self.batch_nums)
                #     self.writer.add_scalar('HITS5-test', hits5_, self.batch_nums)
                #     self.writer.add_scalar('MAP-test', mrr_, self.batch_nums)

            if self.batch_nums % self.log_every == 0:
                # self.save()
                # logging.info('AVG. BATCH_LOSS: {.2f} AT STEP {}'.format(np.mean(losses), self.batch_nums))
                self.writer.add_scalar('Avg_batch_loss', np.mean(losses), self.batch_nums)


            self.batch_nums += 1
            self.scheduler.step()
            if self.batch_nums == self.max_batches:
                self.save()
                break


    def eval(self, mode='dev', meta=False):
        self.matcher.eval()

        symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        if mode == 'dev':
            test_tasks = json.load(open(self.dataset + '/dev_tasks.json'))
        else:
            test_tasks = json.load(open(self.dataset + '/test_tasks.json'))
        
        rel2candidates = self.rel2candidates

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []

        for query_ in test_tasks.keys():

            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []

            candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]
            support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]

            if meta:
                support_left = [self.ent2id[triple[0]] for triple in support_triples]
                support_right = [self.ent2id[triple[2]] for triple in support_triples]
                support_meta = self.get_meta(support_left, support_right)

            support = Variable(torch.LongTensor(support_pairs)).cuda()

            for triple in test_tasks[query_][few:]:
                true = triple[2]
                query_pairs = []
                query_pairs.append([symbol2id[triple[0]], symbol2id[triple[2]]])

                if meta:
                    query_left = []
                    query_right = []
                    query_left.append(self.ent2id[triple[0]])
                    query_right.append(self.ent2id[triple[2]])

                for ent in candidates:
                    if (ent not in self.e1rel_e2[triple[0]+triple[1]]) and ent != true:
                        query_pairs.append([symbol2id[triple[0]], symbol2id[ent]])
                        if meta:
                            query_left.append(self.ent2id[triple[0]])
                            query_right.append(self.ent2id[ent])
                    
                query = Variable(torch.LongTensor(query_pairs)).cuda()

                if meta:
                    query_meta = self.get_meta(query_left, query_right)
                    scores = self.matcher(query, support, query_meta, support_meta)
                    scores.detach()
                    scores = scores.data
                else:
                    scores = self.matcher(query, support)
                    scores.detach()
                    scores = scores.data

                scores = scores.cpu().numpy()
                sort = list(np.argsort(scores))[::-1]
                rank = sort.index(0) + 1
                if rank <= 10:
                    hits10.append(1.0)
                    hits10_.append(1.0)
                else:
                    hits10.append(0.0)
                    hits10_.append(0.0)
                if rank <= 5:
                    hits5.append(1.0)
                    hits5_.append(1.0)
                else:
                    hits5.append(0.0)
                    hits5_.append(0.0)
                if rank <= 1:
                    hits1.append(1.0)
                    hits1_.append(1.0)
                else:
                    hits1.append(0.0)
                    hits1_.append(0.0)
                mrr.append(1.0/rank)
                mrr_.append(1.0/rank)


            logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(query_, np.mean(hits10_), np.mean(hits5_), np.mean(hits1_), np.mean(mrr_)))
            logging.info('Number of candidates: {}, number of text examples {}'.format(len(candidates), len(hits10_)))
            # print query_ + ':'
            # print 'HITS10: ', np.mean(hits10_)
            # print 'HITS5: ', np.mean(hits5_)
            # print 'HITS1: ', np.mean(hits1_)
            # print 'MAP: ', np.mean(mrr_)

        logging.critical('HITS10: {:.3f}'.format(np.mean(hits10)))
        logging.critical('HITS5: {:.3f}'.format(np.mean(hits5)))
        logging.critical('HITS1: {:.3f}'.format(np.mean(hits1)))
        logging.critical('MAP: {:.3f}'.format(np.mean(mrr)))

        self.matcher.train()

        return np.mean(hits10), np.mean(hits5), np.mean(mrr)

    def test_(self):
        self.load()
        logging.info('Pre-trained model loaded')
        self.eval(mode='dev', meta=self.meta)
        self.eval(mode='test', meta=self.meta)

if __name__ == '__main__':
    args = read_options()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # setup random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args)
    if args.test:
        trainer.test_()
    else:
        trainer.train()
    # trainer.eval()