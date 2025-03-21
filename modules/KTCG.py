import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

class Aggregator(nn.Module):

    def __init__(self, n_users, n_virtual, n_iter):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_virtual = n_virtual
        self.n_iter = n_iter
        self.w = torch.nn.Parameter(torch.FloatTensor([0.4, 0.3, 0.3]), requires_grad=True)

    def forward(self, entity_emb, user_emb,
                edge_index, edge_type, adj_mat):
        device = torch.device("cuda:0")

        n_entities = entity_emb.shape[0]
        n_users = self.n_users

        edge_type_uni = torch.unique(edge_type)
        entity_emb_list = []

        user_index, item_index = adj_mat.nonzero()
        user_index = torch.tensor(user_index).type(torch.long).to(device)
        item_index = torch.tensor(item_index).type(torch.long)

        for i in edge_type_uni:
            index = torch.where(edge_type == i)
            index = index[0]
            head, tail = edge_index
            head = head[index]
            tail = tail[index]

            u = None
            neigh_emb = entity_emb[tail]

            for clus_iter in range(self.n_iter):
                if u is None:
                    u = scatter_mean(src=neigh_emb, index=head, dim_size=n_entities, dim=0)
                else:
                    center_emb = u[head]
                    sim = torch.sum(center_emb * neigh_emb, dim=1)
                    n, d = neigh_emb.size()
                    sim = torch.unsqueeze(sim, dim=1)
                    sim.expand(n, d)
                    neigh_emb = sim * neigh_emb
                    u = scatter_mean(src=neigh_emb, index=head, dim_size=n_entities, dim=0)

                if clus_iter < self.n_iter - 1:
                    squash = torch.norm(u, dim=1) ** 2 / (torch.norm(u, dim=1) ** 2 + 1)
                    u = squash.unsqueeze(1) * F.normalize(u, dim=1)
                u += entity_emb
            entity_emb_list.append(u)

        entity_emb_list = torch.stack(entity_emb_list, dim=0)

        item_0 = entity_emb_list[0]
        item_1 = entity_emb_list[1]
        item_2 = entity_emb_list[2]
        w0 = self.w[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        w1 = self.w[1].unsqueeze(dim=-1).unsqueeze(dim=-1)
        w2 = self.w[2].unsqueeze(dim=-1).unsqueeze(dim=-1)
        w_0 = torch.exp(w0) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
        w_1 = torch.exp(w1) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
        w_2 = torch.exp(w2) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
        entity_agg = w_0.mul(item_0) + w_1.mul(item_1) + w_2.mul(item_2)  # entity embedding after aggregate

        u = None
        for clus_iter in range(self.n_iter):
            neigh_emb = entity_emb[item_index]
            if u is None:
                u = scatter_mean(src=neigh_emb, index=user_index, dim_size=n_users, dim=0)
            else:
                center_emb = u[user_index]
                sim = torch.sum(center_emb * neigh_emb, dim=1)
                n, d = neigh_emb.size()
                sim = torch.unsqueeze(sim, dim=1)
                sim.expand(n, d)
                neigh_emb = sim * neigh_emb
                u = scatter_mean(src=neigh_emb, index=user_index, dim_size=n_users, dim=0)

            if clus_iter < self.n_iter - 1:
                squash = torch.norm(u, dim=1) ** 2 / (torch.norm(u, dim=1) ** 2 + 1)
                u = squash.unsqueeze(1) * F.normalize(u, dim=1)
            u += user_emb
        user_agg = u

        return entity_agg, user_agg

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_iter, n_users,
                 n_virtual, n_relations, adj_mat, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.adj_mat = adj_mat
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_virtual = n_virtual
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))
        self.weight = nn.Parameter(weight)

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_virtual=n_virtual, n_iter=n_iter))

        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_emb, entity_emb, latent_emb, edge_index, edge_type,
                adj_mat, interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)

        entity_res_emb = entity_emb
        user_res_emb = user_emb
        cor = 0
        weight = self.weight
        relation_ = torch.mm(weight, latent_emb.t())
        relation_remap = torch.argmax(relation_, dim=1)
        edge_type = relation_remap[edge_type - 1]

        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb,
                                                 edge_index, edge_type, adj_mat)

            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, cor

class Recommender(nn.Module):

    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']
        self.n_nodes = data_config['n_nodes']

        self.decay = args_config.l2
        self.kg_l2loss_lambda = args_config.kg_l2loss_lambda
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_iter = args_config.n_iter
        self.n_virtual = args_config.n_virtual
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)
        self.lightgcn_layer = 2
        self.n_item_layer = 1
        self.alpha = 0.2
        self.fc1 = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
        )

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_virtual, self.emb_size))

        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_iter=self.n_iter,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_virtual=self.n_virtual,
                         adj_mat=self.adj_mat,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, cf_batch):
        user = cf_batch['users']
        item = cf_batch['items']
        labels = cf_batch['labels']
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.latent_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.adj_mat,
                                                     self.interact_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        u_e_1 = user_gcn_emb[user]
        i_e_1 = entity_gcn_emb[item]

        interact_mat_new = self.interact_mat
        indice_old = interact_mat_new._indices()
        value_old = interact_mat_new._values()
        x = indice_old[0, :]
        y = indice_old[1, :]
        x_A = x
        y_A = y + self.n_users
        x_A_T = y + self.n_users
        y_A_T = x
        x_new = torch.cat((x_A, x_A_T), dim=-1)
        y_new = torch.cat((y_A, y_A_T), dim=-1)
        indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
        value_new = torch.cat((value_old, value_old), dim=-1)
        interact_graph = torch.sparse.FloatTensor(indice_new, value_new, torch.Size(
            [self.n_users + self.n_entities, self.n_users + self.n_entities]))
        user_lightgcn_emb, item_lightgcn_emb = self.light_gcn(user_emb, item_emb, interact_graph)

        u_e_2 = user_lightgcn_emb[user]
        i_e_2 = item_lightgcn_emb[item]

        item_1 = item_emb[item]
        user_1 = user_emb[user]

        loss_contrast = self.calculate_loss(i_e_1, i_e_2)
        loss_contrast = loss_contrast + self.calculate_loss(u_e_1, u_e_2)
        loss_contrast_i = loss_contrast + self.calculate_loss_1(item_1, i_e_2)
        loss_contrast_u = loss_contrast + self.calculate_loss_2(user_1, u_e_2)

        u_e = torch.cat((user_1, u_e_1, u_e_2), dim=-1)
        i_e = torch.cat((item_1, i_e_1, i_e_2), dim=-1)

        return self.create_bpr_loss(u_e, i_e, labels, loss_contrast_i, loss_contrast_u)

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.adj_mat,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)[:-1]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def calculate_loss(self, A_embedding, B_embedding):
        self.tau = nn.Parameter(torch.tensor(0.6))
        f = lambda x: torch.exp(x / self.tau)
        A_embedding = self.fc1(A_embedding)
        B_embedding = self.fc1(B_embedding)

        refl_sim = self.sim(A_embedding, A_embedding) / self.tau
        between_sim = self.sim(A_embedding, B_embedding) / self.tau

        refl_log_sum_exp = torch.logsumexp(refl_sim, dim=1)
        between_log_sum_exp = torch.logsumexp(between_sim, dim=1)

        loss_1 = -torch.log(torch.diag(between_sim) - refl_log_sum_exp + between_log_sum_exp)

        ret = loss_1.mean()
        return ret

    def calculate_loss_1(self, A_embedding, B_embedding):

        self.tau = nn.Parameter(torch.tensor(0.6))
        f = lambda x: torch.exp(x / self.tau)
        A_embedding = self.fc2(A_embedding)
        B_embedding = self.fc2(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret

    def calculate_loss_2(self, A_embedding, B_embedding):

        self.tau = nn.Parameter(torch.tensor(0.6))
        f = lambda x: torch.exp(x / self.tau)
        A_embedding = self.fc3(A_embedding)
        B_embedding = self.fc3(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret

    def light_gcn(self, user_embedding, item_embedding, adj):
        ego_embeddings = torch.cat((user_embedding, item_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.lightgcn_layer):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_entities], dim=0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, items, labels, loss_contrast_i, loss_contrast_u):
        batch_size = users.shape[0]
        scores = (items * users).sum(dim=1)
        scores = torch.sigmoid(scores)

        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]

        num_samples = min(positive_scores.size(0), negative_scores.size(0))

        if positive_scores.size(0) > num_samples:
            pos_indices = torch.randperm(positive_scores.size(0))[:num_samples]
            positive_scores = positive_scores[pos_indices]

        if negative_scores.size(0) > num_samples:
            neg_indices = torch.randperm(negative_scores.size(0))[:num_samples]
            negative_scores = negative_scores[neg_indices]

        margin = 1.0
        mar_loss = F.relu(margin - positive_scores + negative_scores).mean()

        regularizer = (torch.norm(users, p=2) ** 2 + torch.norm(items, p=2) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mar_loss + emb_loss + 0.2 * loss_contrast_i + 0.1 * loss_contrast_u, scores, mar_loss, emb_loss

    def _L2_loss_mean(self, x):
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)