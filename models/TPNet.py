import torch
import numpy as np
import torch.nn as nn
from utils.utils import NeighborSampler
import math
from models.modules import TimeEncoder


class RandomProjectionModule(nn.Module):
    def __init__(self, node_num: int, edge_num: int, dim_factor: int, num_layer: int, time_decay_weight: float,
                 device: str, use_matrix: bool, beginning_time: np.float64, not_scale: bool, enforce_dim: int):
        """
        This model maintains a series of temporal walk matrices $A_^(0)(t),A_^(1)(t),...,A^(k)(t)$ through
        random feature propagation, and extract the pairwise features from the obtained random projections.
        :param node_num: int, the number of nodes
        :param edge_num: int, the number of edges
        :param dim_factor: int, the parameter to control the dimension of random projections. Specifically, the
                           dimension of the random projections is set to be dim_factor * log(2*edge_num)
        :param num_layer: int, the max hop of the maintained temporal walk matrices
        :param time_decay_weight: float, the time decay weight (lambda of the original paper)
        :param device: str, torch device
        :param use_matrix: bool, if True, explicitly maintain the temporal walk matrices
        :param beginning_time: np.float64, the earliest time in the given temporal graph
        :param not_scale: bool, if True, the inner product of nodes' random projections will not be scaled
        :param enforce_dim: int, if not -1, explicitly set the dimension of random projections to enforce_dim
        """
        super(RandomProjectionModule, self).__init__()
        self.node_num = node_num
        self.edge_num = edge_num
        if enforce_dim != -1:
            self.dim = enforce_dim
        else:
            self.dim = min(int(math.log(self.edge_num * 2)) * dim_factor, node_num)
        self.num_layer = num_layer
        self.time_decay_weight = time_decay_weight
        self.begging_time = nn.Parameter(torch.tensor(beginning_time), requires_grad=False)
        self.now_time = nn.Parameter(torch.tensor(beginning_time), requires_grad=False)
        self.device = device
        self.random_projections = nn.ParameterList()
        self.use_matrix = use_matrix
        self.node_feature_dim = 128
        self.not_scale = not_scale
        # if use_matrix = True, directly store the temporal walk matrices
        if self.use_matrix:
            self.dim = self.node_num
            for i in range(self.num_layer + 1):
                if i == 0:
                    self.random_projections.append(
                        nn.Parameter(torch.eye(self.node_num), requires_grad=False))
                else:
                    self.random_projections.append(
                        nn.Parameter(torch.zeros_like(self.random_projections[i - 1]), requires_grad=False))
        # otherwise, store the random projection of the temporal walk matrices
        else:
            for i in range(self.num_layer + 1):
                if i == 0:
                    self.random_projections.append(
                        nn.Parameter(torch.normal(0, 1 / math.sqrt(self.dim), (self.node_num, self.dim)),
                                     requires_grad=False))
                else:
                    self.random_projections.append(
                        nn.Parameter(torch.zeros_like(self.random_projections[i - 1]), requires_grad=False))
        self.pair_wise_feature_dim = (2 * self.num_layer + 2) ** 2
        self.mlp = nn.Sequential(nn.Linear(self.pair_wise_feature_dim, self.pair_wise_feature_dim * 4), nn.ReLU(),
                                 nn.Linear(self.pair_wise_feature_dim * 4, self.pair_wise_feature_dim))

    def update(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        updating the temporal walk matrices after observing a batch of interactions.
        :param src_node_ids: np.ndarray, shape (batch,),source node ids
        :param dst_node_ids: np.ndarray, shape (batch,), destination node ids
        :param node_interact_times: np.ndarray, shape (batch,), timestamps of interactions
        """
        src_node_ids = torch.from_numpy(src_node_ids).to(self.device)
        dst_node_ids = torch.from_numpy(dst_node_ids).to(self.device)
        next_time = node_interact_times[-1]
        node_interact_times = torch.from_numpy(node_interact_times).to(dtype=torch.float, device=self.device)
        time_weight = torch.exp(-self.time_decay_weight * (next_time - node_interact_times))[:, None]

        # updating for the current timestamp being moved
        # since the current timestamp will be set to the biggest timestamp in this batch
        # so the random projections should be multiplied by the corresponding time decay weight
        for i in range(1, self.num_layer + 1):
            self.random_projections[i].data = self.random_projections[i].data * np.power(np.exp(
                -self.time_decay_weight * (next_time - self.now_time.cpu().numpy())), i)

        # updating for adding new interactions
        # we use the batch updating schema, where we first computing the influence of each interaction
        # and then aggregate them together
        for i in range(self.num_layer, 0, -1):
            src_update_messages = self.random_projections[i - 1][dst_node_ids] * time_weight
            dst_update_messages = self.random_projections[i - 1][src_node_ids] * time_weight
            self.random_projections[i].scatter_add_(dim=0, index=src_node_ids[:, None].expand(-1, self.dim),
                                                    src=src_update_messages)
            self.random_projections[i].scatter_add_(dim=0, index=dst_node_ids[:, None].expand(-1, self.dim),
                                                    src=dst_update_messages)

        # set current timestamp to the biggest timestamp in this batch
        self.now_time.data = torch.tensor(next_time, device=self.device)

    def get_random_projections(self, node_ids: np.ndarray):
        """
        get the random projections of the give node ids.
        :param node_ids: np.ndarray, shape (batch,)
        :return:
        """
        random_projections = []
        for i in range(self.num_layer + 1):
            random_projections.append(self.random_projections[i][node_ids])
        return random_projections

    def get_pair_wise_feature(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray):
        """
        get pairwise feature for given source nodes and destination nodes.
        :param src_node_ids: np.ndarray, shape (batch,)
        :param dst_node_ids: np.ndarray, shape (batch,)
        :return:
        """
        src_random_projections = torch.stack(self.get_random_projections(src_node_ids), dim=1)
        dst_random_projections = torch.stack(self.get_random_projections(dst_node_ids), dim=1)
        random_projections = torch.cat([src_random_projections, dst_random_projections], dim=1)
        random_feature = torch.matmul(random_projections, random_projections.transpose(1, 2)).reshape(
            len(src_node_ids), -1)
        if self.not_scale:
            return self.mlp(random_feature)
        else:
            random_feature[random_feature < 0] = 0
            random_feature = torch.log(random_feature + 1.0)
            return self.mlp(random_feature)

    def reset_random_projections(self):
        """
        reset the random projections
        """
        for i in range(1, self.num_layer + 1):
            nn.init.zeros_(self.random_projections[i])
        self.now_time.data = self.begging_time.clone()
        if not self.use_matrix:
            nn.init.normal_(self.random_projections[0], mean=0, std=1 / math.sqrt(self.dim))

    def backup_random_projections(self):
        """
        backup the random projections.
        :return: tuple of (now_time,random_projections)
        """
        return self.now_time.clone(), [self.random_projections[i].clone() for i in
                                       range(1, self.num_layer + 1)]

    def reload_random_projections(self, random_projections):
        """
        reload the random projections.
        :param random_projections: tuple of (now_time,random_projections)
        """
        now_time, random_projections = random_projections
        self.now_time.data = now_time.clone()
        for i in range(1, self.num_layer + 1):
            self.random_projections[i].data = random_projections[i - 1].clone()


class TPNet(torch.nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, dropout: float, random_projections: RandomProjectionModule,
                 num_layers: int, num_neighbors: int, device: str):
        """
        Time decay matrix Projection-based graph neural Network for temporal link prediction, named TPNet for short.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param dropout: float, dropout rate
        :param random_projections: RandomProjectionModule, the projected time decay temporal walk matrices
        :param num_layers: int, number of embedding layers
        :param num_neighbors: int, number of sampled neighbors
        :param device: str, device
        """
        super(TPNet, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.dropout = dropout
        self.device = device

        # number of nodes, including the padded node
        self.num_nodes = self.node_raw_features.shape[0]

        self.random_projections = random_projections
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # embedding module
        self.embedding_module = TPNetEmbedding(node_raw_features=self.node_raw_features,
                                               edge_raw_features=self.edge_raw_features,
                                               neighbor_sampler=neighbor_sampler,
                                               time_encoder=self.time_encoder,
                                               node_feat_dim=self.node_feat_dim,
                                               edge_feat_dim=self.edge_feat_dim,
                                               time_feat_dim=self.time_feat_dim,
                                               num_layers=num_layers,
                                               num_neighbors=num_neighbors,
                                               dropout=self.dropout,
                                               random_projections=self.random_projections)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings.
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        node_embeddings = self.embedding_module.compute_node_temporal_embeddings(
            node_ids=np.concatenate([src_node_ids, dst_node_ids]),
            src_node_ids=np.tile(src_node_ids, 2),
            dst_node_ids=np.tile(dst_node_ids, 2),
            node_interact_times=np.tile(node_interact_times, 2))
        src_node_embeddings, dst_node_embeddings = node_embeddings[:len(src_node_ids)], node_embeddings[
                                                                                        len(src_node_ids):]
        return src_node_embeddings, dst_node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling).
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.embedding_module.neighbor_sampler = neighbor_sampler
        if self.embedding_module.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.embedding_module.neighbor_sampler.seed is not None
            self.embedding_module.neighbor_sampler.reset_random_state()


class TPNetEmbedding(nn.Module):
    def __init__(self, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor,
                 neighbor_sampler: NeighborSampler,
                 time_encoder: nn.Module, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_layers: int, num_neighbors: int, dropout: float, random_projections: RandomProjectionModule):
        """
        Embedding module of TPNet, which utilizes a multi-layer MLP-Mixer as its backbone.
        :param node_raw_features: Tensor, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: Tensor, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_encoder: TimeEncoder
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim:  int, dimension of time features (encodings)
        :param num_layers: int, number of MLP-Mixer layers
        :param dropout: float, dropout rate
        """
        super(TPNetEmbedding, self).__init__()

        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.neighbor_sampler = neighbor_sampler
        self.time_encoder = time_encoder
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.random_projections = random_projections
        if self.random_projections is None:
            self.random_feature_dim = 0
        else:
            self.random_feature_dim = self.random_projections.pair_wise_feature_dim * 2
        self.projection_layer = nn.Sequential(
            nn.Linear(node_feat_dim + edge_feat_dim + time_feat_dim + self.random_feature_dim, self.node_feat_dim * 2),
            nn.ReLU(), nn.Linear(self.node_feat_dim * 2, self.node_feat_dim))
        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=self.num_neighbors, num_channels=self.node_feat_dim,
                     token_dim_expansion_factor=0.5,
                     channel_dim_expansion_factor=4.0, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, src_node_ids: np.ndarray,
                                         dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        given memory, node ids node_ids, and the corresponding time node_interact_times, return the temporal embeddings.
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        """

        device = self.node_raw_features.device
        # get temporal neighbors, including neighbor ids, edge ids and time information
        # neighbor_node_ids ndarray, shape (batch_size, num_neighbors)
        # neighbor_edge_ids ndarray, shape (batch_size, num_neighbors)
        # neighbor_times ndarray, shape (batch_size, num_neighbors)
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.num_neighbors)
        # get node features, shape (batch,num_neighbors,node_feat_dim)
        neighbor_node_features = self.node_raw_features[torch.from_numpy(neighbor_node_ids)]
        neighbor_delta_times = torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(device)
        # scale the delta times
        neighbor_delta_times = torch.log(neighbor_delta_times + 1.0)
        # get time encoding, shape (batch,num_neighbors, time_feat_dim)
        neighbor_time_features = self.time_encoder(neighbor_delta_times)
        # get edge features, shape (batch,num_neighors,edge_feat_dim)
        neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]

        # assign relative encodings for neighbor nodes
        # given a source node u, a destination ndoe v, and a target node w (neighbor of u or v)
        # its relative encoding is [r_{w|u},r_{w|v}], where r_{w|u}/r_{w|v} is the pairwise feature
        # given by the calling the get_pair_wise_feature(w,u)/get_pair_wise_feature(w,v) of the RandomProjectionModule
        if self.random_projections is not None:
            # [2*batch*num_neighbors,random_feature_dim]
            concat_neighbor_random_features = self.random_projections.get_pair_wise_feature(
                src_node_ids=np.tile(neighbor_node_ids.reshape(-1), 2),
                dst_node_ids=np.concatenate(
                    [np.repeat(src_node_ids, self.num_neighbors), np.repeat(dst_node_ids, self.num_neighbors)]))
            # [batch,num_neighbors,random_feature_dim*2]
            neighbor_random_features = torch.cat(
                [concat_neighbor_random_features[:len(node_ids) * self.num_neighbors],
                 concat_neighbor_random_features[len(node_ids) * self.num_neighbors:]],
                dim=1).reshape(len(node_ids), self.num_neighbors, -1)
            neighbor_combine_features = torch.cat(
                [neighbor_node_features, neighbor_time_features, neighbor_edge_features, neighbor_random_features],
                dim=2)
        else:
            neighbor_combine_features = torch.cat(
                [neighbor_node_features, neighbor_time_features, neighbor_edge_features], dim=2)

        # shape (batch, num_neighbors, node_feat_dim)
        embeddings = self.projection_layer(neighbor_combine_features)
        # mask the pad nodes (i.e., id = 0)
        embeddings.masked_fill(torch.from_numpy(neighbor_node_ids == 0)[:, :, None].to(device), 0)
        for mlp_mixer in self.mlp_mixers:
            embeddings = mlp_mixer(embeddings)
        # shape (batch, node_feat_dim)
        embeddings = torch.mean(embeddings, dim=1)

        return embeddings


class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)


class MLPMixer(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.0):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(MLPMixer, self).__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,
                                                dropout=dropout)

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels,
                                                  dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)

    def forward(self, input_tensor: torch.Tensor):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        # mix tokens
        # Tensor, shape (batch_size, num_channels, num_tokens)
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor

        # mix channels
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + output_tensor

        return output_tensor
