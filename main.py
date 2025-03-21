import random
import torch
import numpy as np
from time import time

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from prettytable import PrettyTable
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity

from utils.parser import parse_args
from utils.data_loader import load_data
from modules.KTCG import Recommender
from utils.evaluate import test
from utils.helper import early_stopping

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end):
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, triplets, relation_dict, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in test_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in test_cf], np.int32))

    """kg data"""
    train_kg_pairs = torch.LongTensor(np.array([[kg[0], kg[1], kg[2]] for kg in triplets], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    best_epoch = 0
    should_stop = False

    print("start training ...")
    for epoch in range(args.epoch):
        """training CF"""

        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf = train_cf[index]

        """ training """
        """ train cf """
        mf_loss_total, s = 0, 0
        train_cf_s = time()
        model.train()
        flag = 'cf'
        while s + args.batch_size <= len(train_cf):
            cf_batch = get_feed_dict(train_cf, s, s + args.batch_size)
            mf_loss, _, _, _ = model(cf_batch)

            optimizer.zero_grad()
            mf_loss.backward()
            optimizer.step()

            mf_loss_total += mf_loss.item()
            s += args.batch_size

        train_cf_e = time()

        if epoch % 10 == 9 or epoch == 1:
            """testing"""
            test_s_t = time()
            model.eval()
            ret = test(model, user_dict, n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "testing time", "recall", "ndcg", "precision",
                                     "hit_ratio", "auc", "f1"]
            train_res.add_row(
                [epoch, train_cf_e - train_cf_s, test_e_t - test_s_t, ret['recall'], ret['ndcg'], ret['precision'],
                 ret['hit_ratio'], ret['auc'], ret['f1']]
            )
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
            cur_best_pre_0, stopping_step, should_stop, best_epoch = early_stopping(ret['recall'][3], cur_best_pre_0,
                                                                                    stopping_step, best_epoch, epoch,
                                                                                    expected_order='acc',
                                                                                    flag_step=10)
            if should_stop:
                break

        else:
            print('using time %.4f, training loss at epoch %d: %.4f' % (train_cf_e - train_cf_s, epoch, mf_loss_total))

    print('stopping at %d, recall@20:%.4f' % (epoch, ret['recall'][3]))
    print('the best epoch is at %d, recall@20:%.4f' % (best_epoch, cur_best_pre_0))