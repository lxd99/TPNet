import torch
import torch.nn as nn
import numpy as np
import enum


class NAT(torch.nn.Module):
    def __init__(self, n_feat: np.ndarray, e_feat: np.ndarray, time_dim: int = 2, num_neighbors: list = ['1', '32'],
                 dropout: float = 0.1, n_hops: int = 2, ngh_dim: int = 4, device: str = None):
        """
        NAT module. The original code can be found at https://github.com/Graph-COM/Neighborhood-Aware-Temporal-Network
        :param n_feat: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param e_feat: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param time_dim: int, dimension of time features
        :param num_neighbors: int, number of neighbors for different hop of ncahce
        :param dropout: float, dropout
        :param n_hops: int, maximum hop of neighbors
        :param ngh_dim: int, dimension of the ncache value
        :param device: str, pytorch device
        """
        super(NAT, self).__init__()
        self.dropout = dropout
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)
        self.feat_dim = self.n_feat_th.shape[1]  # node feature dimension
        self.e_feat_dim = self.e_feat_th.shape[1]  # edge feature dimension
        self.time_dim = time_dim  # default to be time feature dimension
        self.self_dim = self.n_feat_th.shape[1]
        self.ngh_dim = ngh_dim
        # embedding layers and encoders
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.time_encoder = self.init_time_encoder()  # fourier
        self.device = device

        self.pos_dim = 16
        self.trainable_embedding = nn.Embedding(num_embeddings=64, embedding_dim=self.pos_dim)  # position embedding

        # final projection layer
        self.num_neighbors = num_neighbors
        self.n_hops = n_hops
        self.ngh_id_idx = 0
        self.e_raw_idx = 1
        self.ts_raw_idx = 2
        self.num_raw = 3

        self.ngh_rep_idx = [self.num_raw, self.num_raw + self.ngh_dim]

        self.memory_dim = self.num_raw + self.ngh_dim

        self.attn_dim = self.feat_dim + self.ngh_dim + self.pos_dim
        self.gat = GAT(1, [2], [self.attn_dim, self.feat_dim], add_skip_connection=False, bias=True,
                       dropout=dropout, log_attention_weights=False)
        self.total_nodes = n_feat.shape[0]
        self.replace_prob = 0.9
        self.self_rep_linear = nn.Linear(self.self_dim + self.time_dim + self.e_feat_dim, self.self_dim, bias=False)
        self.ngh_rep_linear = nn.Linear(self.self_dim + self.time_dim + self.e_feat_dim, self.ngh_dim, bias=False)
        self.self_aggregator = self.init_self_aggregator()  # RNN
        self.ngh_aggregator = self.init_ngh_aggregator()  # RNN

        self.neighborhood_store = nn.ParameterList()
        for i in self.num_neighbors:
            max_e_idx = self.total_nodes * i
            raw_store = torch.zeros(max_e_idx, self.num_raw)
            hidden_store = torch.empty(max_e_idx, self.ngh_dim)
            ngh_store = torch.cat((raw_store, nn.init.xavier_uniform_(hidden_store)), -1)
            self.neighborhood_store.append(nn.Parameter(ngh_store, requires_grad=False))
        self.self_rep = nn.Parameter(torch.zeros(self.total_nodes, self.self_dim), requires_grad=False)
        self.prev_raw = nn.Parameter(torch.zeros(self.total_nodes, 3), requires_grad=False)
        self.generator = torch.Generator(device=self.device)

    def set_seed(self, seed):
        self.seed = seed
        self.generator.manual_seed(seed)

    def reset_random_state(self):
        self.generator.manual_seed(self.seed)

    def init_ncache(self):
        for id, i in enumerate(self.num_neighbors):
            max_e_idx = self.total_nodes * i
            raw_store = torch.zeros(max_e_idx, self.num_raw)
            hidden_store = torch.empty(max_e_idx, self.ngh_dim)
            ngh_store = torch.cat((raw_store, nn.init.xavier_uniform_(hidden_store)), -1).to(self.device)
            self.neighborhood_store[id].data = ngh_store
        nn.init.zeros_(self.self_rep)
        nn.init.zeros_(self.prev_raw)

    def backup_ncache(self):
        return [x.clone() for x in self.neighborhood_store], self.self_rep.clone(), self.prev_raw.clone()

    def reload_ncache(self, data):
        neighbor_store, self_rep, prev_raw = data
        for i in range(len(self.neighborhood_store)):
            self.neighborhood_store[i].data = neighbor_store[i].clone()
        self.self_rep.data = self_rep.clone()
        self.prev_raw.data = prev_raw.clone()

    def set_device(self, device):
        self.device = device

    def position_bits(self, bs, hop):
        return torch.ones(bs * self.num_neighbors[hop], device=self.device, dtype=torch.float) * np.power(
            torch.tensor(2.0, dtype=torch.float), hop)

    def compute_edge_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times, edge_ids,
                                         edges_are_positive=False):
        batch_size = len(src_node_ids)

        # Move data to the GPU
        src_node_ids = torch.from_numpy(src_node_ids).to(dtype=torch.long, device=self.device)
        dst_node_ids = torch.from_numpy(dst_node_ids).to(dtype=torch.long, device=self.device)

        node_ids = torch.cat((src_node_ids, dst_node_ids), 0)
        node_interact_times = torch.from_numpy(node_interact_times).to(dtype=torch.float, device=self.device)
        batch_idx = torch.arange(batch_size * 2, device=self.device)

        self.neighborhood_store[0][node_ids, 0] = node_ids.float()
        # n_id is the node idx of neighbors of query node
        # dense_idx is the position of each neighbors in the batch*nngh tensor
        # sprase_idx is a tensor of batch idx repeated with ngh_n timesfor each node

        h0_pos_bit = self.position_bits(2 * batch_size, hop=0)
        # [3*batch*num_neighbor_0,raw_dim+ngh_dim]
        updated_mem_h0 = self.batch_fetch_ncaches(node_ids, node_interact_times.repeat(2), hop=0)
        # [batch*layer_repeat,raw_dim+ngh_dim + 1]
        updated_mem_h0_with_pos = torch.cat((updated_mem_h0, h0_pos_bit.unsqueeze(1)), -1)
        feature_dim = self.memory_dim + 1
        updated_mem = updated_mem_h0_with_pos.view(2 * batch_size, self.num_neighbors[0], -1)
        updated_mem_h1 = None
        if self.n_hops > 0:
            h1_pos_bit = self.position_bits(2 * batch_size, hop=1)
            updated_mem_h1 = self.batch_fetch_ncaches(node_ids, node_interact_times.repeat(2), hop=1)
            updated_mem_h1_with_pos = torch.cat((updated_mem_h1, h1_pos_bit.unsqueeze(1)), -1)
            updated_mem = torch.cat((
                updated_mem,
                updated_mem_h1_with_pos.view(2 * batch_size, self.num_neighbors[1], -1)), 1)
        if self.n_hops > 1:
            # second-hop N-cache access
            h2_pos_bit = self.position_bits(2 * batch_size, hop=2)
            updated_mem_h2 = torch.cat(
                (self.batch_fetch_ncaches(node_ids, node_interact_times.repeat(3), hop=2), h2_pos_bit.unsqueeze(1)), -1)
            updated_mem = torch.cat((updated_mem, updated_mem_h2.view(2 * batch_size, self.num_neighbors[2], -1)), 1)

        updated_mem = updated_mem.view(-1, feature_dim)
        ngh_id = updated_mem[:, self.ngh_id_idx].long()
        ngh_exists = torch.nonzero(ngh_id, as_tuple=True)[0]
        ngh_count = torch.count_nonzero(ngh_id.view(2, batch_size, -1), dim=-1)

        ngh_id = ngh_id.index_select(0, ngh_exists)
        updated_mem = updated_mem.index_select(0, ngh_exists)
        src_ngh_n_th, tgt_ngh_n_th = ngh_count[0], ngh_count[1]
        ngh_n_th = torch.cat((src_ngh_n_th, tgt_ngh_n_th), 0)
        sparse_idx = torch.repeat_interleave(batch_idx, ngh_n_th).long()
        src_nghs = torch.sum(src_ngh_n_th)
        tgt_nghs = torch.sum(tgt_ngh_n_th)

        node_features = self.node_raw_embed(ngh_id)

        pos_raw = updated_mem[:, -1]
        src_pos_raw = pos_raw[0:src_nghs]
        # for the target nodes, shift all the bits by 3 to differentiate from the source nodes
        tgt_pos_raw = pos_raw[src_nghs:src_nghs + tgt_nghs].to(torch.float) * np.power(
            torch.tensor(2.0, dtype=torch.float), 3)
        pos_raw = torch.cat((src_pos_raw, tgt_pos_raw), -1)
        hidden_states = torch.cat(
            (node_features, updated_mem[:, self.ngh_rep_idx[0]:self.ngh_rep_idx[1]], pos_raw.unsqueeze(1)), -1)

        src_prev_f = hidden_states[0:src_nghs]
        tgt_prev_f = hidden_states[src_nghs:src_nghs + tgt_nghs]

        src_ngh_id = ngh_id[0:src_nghs]
        tgt_ngh_id = ngh_id[src_nghs:src_nghs + tgt_nghs]
        src_sparse_idx = sparse_idx[0:src_nghs]
        tgt_sparse_idx = sparse_idx[src_nghs:src_nghs + tgt_nghs] - batch_size

        # joint features construction
        # flatten the neighbor nodes and record its basic info
        joint_p, ngh_and_batch_id_p = self.get_joint_feature(src_sparse_idx, tgt_sparse_idx, src_ngh_id, tgt_ngh_id,
                                                             src_prev_f, tgt_prev_f)
        features = self.get_position_encoding(joint_p)

        src_self_rep = self.updated_self_rep(src_node_ids)
        tgt_self_rep = self.updated_self_rep(dst_node_ids)

        edge_embeddings = self.attn_aggregate(ngh_and_batch_id_p, features, batch_size,
                                              src_self_rep, tgt_self_rep)
        if edges_are_positive:
            edge_ids = torch.from_numpy(edge_ids).to(dtype=torch.long, device=self.device)
            self.self_rep[src_node_ids] = src_self_rep.detach()
            self.self_rep[dst_node_ids] = tgt_self_rep.detach()

            self.prev_raw[src_node_ids] = torch.stack([dst_node_ids, edge_ids, node_interact_times], dim=1)
            self.prev_raw[dst_node_ids] = torch.stack([src_node_ids, edge_ids, node_interact_times], dim=1)

            # N-cache update
            self.update_memory(src_node_ids, dst_node_ids, edge_ids, node_interact_times, updated_mem_h0,
                               updated_mem_h1,
                               batch_size)
        return edge_embeddings

    def get_position_encoding(self, joint):
        if self.pos_dim == 0:
            return joint[:, :-1]
        pos_raw = joint[:, -1]
        pos_encoding = self.trainable_embedding(pos_raw.long())
        return torch.cat((joint[:, :-1], pos_encoding), -1)

    def updated_self_rep(self, node_id):
        self_store = self.prev_raw[node_id]
        oppo_id = self_store[:, self.ngh_id_idx].long()
        e_raw = self_store[:, self.e_raw_idx].long()
        ts_raw = self_store[:, self.ts_raw_idx]
        e_feat = self.edge_raw_embed(e_raw)
        ts_feat = self.time_encoder(ts_raw)
        prev_self_rep = self.self_rep[node_id]
        prev_oppo_rep = self.self_rep[oppo_id]
        updated_self_rep = self.self_aggregator(self.self_rep_linear(torch.cat((prev_oppo_rep, e_feat, ts_feat), -1)),
                                                prev_self_rep)
        return updated_self_rep

    def update_memory(self, src_th, tgt_th, e_idx_th, cut_time_th, updated_mem_h0, updated_mem_h1, batch_size):
        ori_idx = torch.cat((src_th, tgt_th), 0)
        cut_time_th = cut_time_th.repeat(2)
        opp_th = torch.cat((tgt_th, src_th), 0)
        e_idx_th = e_idx_th.repeat(2)
        # Update neighbors
        batch_id = torch.arange(batch_size * 2, device=self.device)
        if self.n_hops > 0:
            # [2*batch*num_neighbor_1,dim]
            updated_mem_h1 = updated_mem_h1.detach()[:2 * batch_size * self.num_neighbors[1]]
            # Update second hop neighbors
            if self.n_hops > 1:
                ngh_h1_id = updated_mem_h1[:, self.ngh_id_idx].long()
                ngh_exists = torch.nonzero(ngh_h1_id, as_tuple=True)[0]
                updated_mem_h2 = updated_mem_h1.index_select(0, ngh_exists)
                ngh_count = torch.count_nonzero(ngh_h1_id.view(2 * batch_size, self.num_neighbors[1]), dim=-1)
                opp_expand_th = torch.repeat_interleave(opp_th, ngh_count, dim=0)
                self.update_ncaches(opp_expand_th, updated_mem_h2, 2)
            updated_mem_h1 = updated_mem_h1[(batch_id * self.num_neighbors[1] + self.ncache_hash(opp_th, 1))]
            # oppt previous is in ori
            ngh_id_is_match = (updated_mem_h1[:, self.ngh_id_idx] == opp_th).unsqueeze(1).repeat(1, self.memory_dim)
            updated_mem_h1 = updated_mem_h1 * ngh_id_is_match

            candidate_ncaches = torch.cat((opp_th.unsqueeze(1), e_idx_th.unsqueeze(1), cut_time_th.unsqueeze(1),
                                           updated_mem_h1[:, self.ngh_rep_idx[0]:self.ngh_rep_idx[1]]), -1)
            self.update_ncaches(ori_idx, candidate_ncaches, 1)
        # Update self
        updated_mem_h0 = updated_mem_h0.detach()[:batch_size * self.num_neighbors[0] * 2]
        candidate_ncaches = torch.cat((ori_idx.unsqueeze(1), e_idx_th.unsqueeze(1), cut_time_th.unsqueeze(1),
                                       updated_mem_h0[:, self.ngh_rep_idx[0]:self.ngh_rep_idx[1]]), -1)
        self.update_ncaches(ori_idx, candidate_ncaches, 0)

    def ncache_hash(self, ngh_id, hop):
        ngh_id = ngh_id.long()
        return ((ngh_id * (self.seed % 100) + ngh_id * ngh_id * ((self.seed % 100) + 1)) % self.num_neighbors[
            hop]).int()

    def update_ncaches(self, self_id, candidate_ncaches, hop):
        if self.num_neighbors[hop] == 0:
            return
        ngh_id = candidate_ncaches[:, self.ngh_id_idx]
        idx = self_id * self.num_neighbors[hop] + self.ncache_hash(ngh_id, hop)
        is_occupied = torch.logical_and(self.neighborhood_store[hop][idx, self.ngh_id_idx] != 0,
                                        self.neighborhood_store[hop][idx, self.ngh_id_idx] != ngh_id)
        should_replace = (is_occupied * torch.rand(is_occupied.shape[0], device=self.device,
                                                   generator=self.generator)) < self.replace_prob
        idx *= should_replace
        idx *= ngh_id != 0
        self.neighborhood_store[hop][idx] = candidate_ncaches

    def get_joint_neighborhood(self, src_sparse_idx, tgt_sparse_idx, src_n_id, tgt_n_id, src_hidden, tgt_hidden):
        sparse_idx = torch.cat((src_sparse_idx, tgt_sparse_idx), 0)
        n_id = torch.cat((src_n_id, tgt_n_id), 0)
        all_hidden = torch.cat((src_hidden, tgt_hidden), 0)
        feat_dim = src_hidden.shape[-1]
        key = torch.cat((sparse_idx.unsqueeze(1), n_id.unsqueeze(1)), -1)  # tuple of (idx in the current batch, n_id)
        unique, inverse_idx = key.unique(return_inverse=True, dim=0)
        # SCATTER ADD FOR TS WITH INV IDX
        relative_ts = torch.zeros(unique.shape[0], feat_dim, device=self.device)
        relative_ts.scatter_add_(0, inverse_idx.unsqueeze(1).repeat(1, feat_dim), all_hidden)
        relative_ts = relative_ts.index_select(0, inverse_idx)
        assert (relative_ts.shape[0] == sparse_idx.shape[0] == all_hidden.shape[0])

        return relative_ts

    def get_joint_feature(self, src_sparse_idx, tgt_sparse_idx, src_n_id, tgt_n_id, src_hidden, tgt_hidden):
        # compute the Q_{u,v} in the paper
        sparse_idx = torch.cat((src_sparse_idx, tgt_sparse_idx), 0)
        n_id = torch.cat((src_n_id, tgt_n_id), 0)
        all_hidden = torch.cat((src_hidden, tgt_hidden), 0)
        feat_dim = src_hidden.shape[-1]
        key = torch.cat((n_id.unsqueeze(1), sparse_idx.unsqueeze(1)), -1)  # tuple of (idx in the current batch, n_id)
        unique, inverse_idx = key.unique(return_inverse=True, dim=0)
        # SCATTER ADD FOR TS WITH INV IDX
        relative_ts = torch.zeros(unique.shape[0], feat_dim, device=self.device)
        relative_ts.scatter_add_(0, inverse_idx.unsqueeze(1).repeat(1, feat_dim), all_hidden)
        return relative_ts, unique

    def batch_fetch_ncaches(self, ori_idx, curr_time, hop):
        ngh = self.neighborhood_store[hop].view(self.total_nodes, self.num_neighbors[hop], self.memory_dim)[
            ori_idx].view(ori_idx.shape[0] * (self.num_neighbors[hop]), self.memory_dim)
        curr_time = curr_time.repeat_interleave(self.num_neighbors[hop])
        ngh_id = ngh[:, self.ngh_id_idx].long()
        ngh_e_raw = ngh[:, self.e_raw_idx].long()
        ngh_ts_raw = ngh[:, self.ts_raw_idx]
        prev_ngh_rep = ngh[:, self.ngh_rep_idx[0]:self.ngh_rep_idx[1]]
        e_feat = self.edge_raw_embed(ngh_e_raw)
        ts_feat = self.time_encoder(ngh_ts_raw)
        ngh_self_rep = self.self_rep[ngh_id]
        updated_self_rep = self.ngh_aggregator(self.ngh_rep_linear(torch.cat((ngh_self_rep, e_feat, ts_feat), -1)),
                                               prev_ngh_rep)
        updated_self_rep *= (ngh_ts_raw != 0).unsqueeze(1).repeat(1, self.ngh_dim)
        ori_idx = torch.repeat_interleave(ori_idx, self.num_neighbors[hop])
        # msk = ngh_ts_raw <= curr_time
        updated_mem = torch.cat((ngh[:, :self.num_raw], updated_self_rep), -1)
        # updated_mem *= msk.unsqueeze(1).repeat(1, self.memory_dim)
        return updated_mem

    def attn_aggregate(self, edge_idx, feat, bs, src_self_rep=None, tgt_self_rep=None):
        edge_idx = edge_idx.T
        edge_emb, _, attn_score = self.gat((feat, edge_idx.long(), bs))
        edge_emb = torch.cat((edge_emb, src_self_rep, tgt_self_rep), -1)
        return edge_emb

    def init_time_encoder(self):
        return TimeEncode(expand_dim=self.time_dim)

    def init_self_aggregator(self):
        return FeatureEncoderGRU(self.self_dim, self.self_dim, self.dropout)

    def init_ngh_aggregator(self):
        return FeatureEncoderGRU(self.ngh_dim, self.ngh_dim, self.dropout)


class FeatureEncoderGRU(torch.nn.Module):
    def __init__(self, input_dim, ngh_dim, dropout_p=0.1):
        super(FeatureEncoderGRU, self).__init__()
        self.gru = nn.GRUCell(input_dim, ngh_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.ngh_dim = ngh_dim

    def forward(self, input_features, hidden_state, use_dropout=False):
        encoded_features = self.gru(input_features, hidden_state)
        if use_dropout:
            encoded_features = self.dropout(encoded_features)

        # return input_features
        return encoded_features


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()

        self.time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts):
        # ts: [N, 1]
        batch_size = ts.size(0)

        ts = ts.view(batch_size, 1)  # [N, 1]
        map_ts = ts * self.basis_freq.view(1, -1)  # [N, time_dim]
        map_ts += self.phase.view(1, -1)  # [N, time_dim]
        harmonic = torch.cos(map_ts)

        # return torch.zeros_like(ts)
        return harmonic  # self.dense(harmonic)


class LayerType(enum.Enum):
    IMP1 = 0,
    IMP2 = 1,
    IMP3 = 2


class GAT(torch.nn.Module):
    """
    I've added 3 GAT implementations - some are conceptually easier to understand some are more efficient.
    The most interesting and hardest one to understand is implementation #3.
    Imp1 and imp2 differ in subtle details but are basically the same thing.
    Tip on how to approach this:
        understand implementation 2 first, check out the differences it has with imp1, and finally tackle imp #3.
    """

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP3, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        GATLayer = get_layer_type(layer_type)  # fetch one of 3 available implementations
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i + 1],
                num_of_heads=num_heads_per_layer[i + 1],
                concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        return self.gat_net(data)


class GATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.
    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, layer_type, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        if layer_type == LayerType.IMP1:
            # Experimenting with different options to see what is faster (tip: focus on 1 implementation at a time)
            self.proj_param = nn.Parameter(torch.Tensor(num_of_heads, num_in_features, num_out_features))
        else:
            # You can treat this one matrix as num_of_heads independent W matrices
            self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if layer_type == LayerType.IMP1:  # simple reshape in the case of implementation 1
            self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
            self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        # if add_skip_connection:
        self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        # else:
        #     self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params(layer_type)

    def init_params(self, layer_type):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.
        """
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads,
                                                                             self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class GATLayerImp3(GATLayer):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric
    But, it's hopefully much more readable! (and of similar performance)
    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3
    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0  # node dimension/axis
    head_dim = 1  # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):
        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP3, concat, activation,
                         dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #

        # in_nodes_features, self_nodes_features, edge_index, num_of_nodes = data  # unpack data
        in_nodes_features, edge_index, num_of_nodes = data  # unpack data
        # num_of_nodes = in_nodes_features.shape[self.nodes_dim]

        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'
        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)
        # self_nodes_features = self.dropout(self_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        # self_nodes_features_proj = self.linear_proj(self_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well
        # self_nodes_features_proj = self.dropout(self_nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # nodes_features_proj = nodes_features_proj * self_nodes_features_proj
        # score = (nodes_features_proj * self_nodes_features_proj).sum(dim=-1)
        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        # scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = scores_source, scores_target, nodes_features_proj
        # source_lifted, nodes_features_proj_lifted = score, nodes_features_proj
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)
        # scores_per_edge = self.leakyReLU(source_lifted)

        # shape = (E, NH, 1)
        # attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        # attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted
        # nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # skip connection
        # nodes_features_proj_lifted_weighted += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index,
                                                      in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #
        out_nodes_features = self.skip_concat_bias(None, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index, None)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.
        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:
        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index,
                                                                                num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim],
                                                        nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
        # scatter_mean(all_hidden, inverse_idx.unsqueeze(1).repeat(1,self.caw_dim), out=relative_ts, dim=0)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)


class GATLayerImp2(GATLayer):
    """
        Implementation #2 was inspired by the official GAT implementation: https://github.com/PetarV-/GAT
        It's conceptually simpler than implementation #3 but computationally much less efficient.
        Note: this is the naive implementation not the sparse one and it's only suitable for a transductive setting.
        It would be fairly easy to make it work in the inductive setting as well but the purpose of this layer
        is more educational since it's way less efficient than implementation 3.
    """

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):
        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP2, concat, activation,
                         dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization (using linear layer instead of matmul as in imp1)
        #

        in_nodes_features, connectivity_mask = data  # unpack data
        num_of_nodes = in_nodes_features.shape[0]
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation (using sum instead of bmm + additional permute calls - compared to imp1)
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1)
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)

        # src shape = (NH, N, 1) and trg shape = (NH, 1, N)
        scores_source = scores_source.transpose(0, 1)
        scores_target = scores_target.permute(1, 2, 0)

        # shape = (NH, N, 1) + (NH, 1, N) -> (NH, N, N) with the magic of automatic broadcast <3
        # In Implementation 3 we are much smarter and don't have to calculate all NxN scores! (only E!)
        # Tip: it's conceptually easier to understand what happens here if you delete the NH dimension
        all_scores = self.leakyReLU(scores_source + scores_target)
        # connectivity mask will put -inf on all locations where there are no edges, after applying the softmax
        # this will result in attention scores being computed only for existing edges
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)

        #
        # Step 3: Neighborhood aggregation (same as in imp1)
        #

        # batch matrix multiply, shape = (NH, N, N) * (NH, N, FOUT) -> (NH, N, FOUT)
        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))

        # Note: watch out here I made a silly mistake of using reshape instead of permute thinking it will
        # end up doing the same thing, but it didn't! The acc on Cora didn't go above 52%! (compared to reported ~82%)
        # shape = (N, NH, FOUT)
        out_nodes_features = out_nodes_features.permute(1, 0, 2)

        #
        # Step 4: Residual/skip connections, concat and bias (same as in imp1)
        #

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        return (out_nodes_features, connectivity_mask)


class GATLayerImp1(GATLayer):
    """
        This implementation is only suitable for a transductive setting.
        It would be fairly easy to make it work in the inductive setting as well but the purpose of this layer
        is more educational since it's way less efficient than implementation 3.
    """

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):
        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP1, concat, activation,
                         dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features, connectivity_mask = data  # unpack data
        num_of_nodes = in_nodes_features.shape[0]
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (1, N, FIN) * (NH, FIN, FOUT) -> (NH, N, FOUT) where NH - number of heads, FOUT num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = torch.matmul(in_nodes_features.unsqueeze(0), self.proj_param)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # batch matrix multiply, shape = (NH, N, FOUT) * (NH, FOUT, 1) -> (NH, N, 1)
        scores_source = torch.bmm(nodes_features_proj, self.scoring_fn_source)
        scores_target = torch.bmm(nodes_features_proj, self.scoring_fn_target)

        # shape = (NH, N, 1) + (NH, 1, N) -> (NH, N, N) with the magic of automatic broadcast <3
        # In Implementation 3 we are much smarter and don't have to calculate all NxN scores! (only E!)
        # Tip: it's conceptually easier to understand what happens here if you delete the NH dimension
        all_scores = self.leakyReLU(scores_source + scores_target.transpose(1, 2))
        # connectivity mask will put -inf on all locations where there are no edges, after applying the softmax
        # this will result in attention scores being computed only for existing edges
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)

        #
        # Step 3: Neighborhood aggregation
        #

        # shape = (NH, N, N) * (NH, N, FOUT) -> (NH, N, FOUT)
        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj)

        # shape = (N, NH, FOUT)
        out_nodes_features = out_nodes_features.transpose(0, 1)

        #
        # Step 4: Residual/skip connections, concat and bias (same across all the implementations)
        #

        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        return (out_nodes_features, connectivity_mask)


#
# Helper functions
#
def get_layer_type(layer_type):
    assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'

    if layer_type == LayerType.IMP1:
        return GATLayerImp1
    elif layer_type == LayerType.IMP2:
        return GATLayerImp2
    elif layer_type == LayerType.IMP3:
        return GATLayerImp3
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')
