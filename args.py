import argparse

def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NELL", type=str)
    parser.add_argument("--embed_dim", default=100, type=int)
    parser.add_argument("--few", default=1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--neg_num", default=1, type=int)
    parser.add_argument("--random_embed", action='store_true')
    parser.add_argument("--train_few", default=1, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--margin", default=5.0, type=float)
    parser.add_argument("--max_batches", default=1000000, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--process_steps", default=2, type=int)
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--eval_every", default=10000, type=int)
    parser.add_argument("--fine_tune", action='store_true')
    parser.add_argument("--aggregate", default='max', type=str)
    parser.add_argument("--max_neighbor", default=200, type=int)
    parser.add_argument("--no_meta", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--grad_clip", default=5.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--embed_model", default='ComplEx', type=str)
    parser.add_argument("--prefix", default='intial', type=str)
    parser.add_argument("--seed", default='19940419', type=int)

    args = parser.parse_args()
    # args.embed_path = args.dataset + '/symbol2vec.vec'
    args.save_path = 'models/' + args.prefix

    print("------HYPERPARAMETERS-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    print("----------------------------")

    return args

