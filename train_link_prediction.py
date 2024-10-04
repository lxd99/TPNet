import logging
import time
import sys
import os
from pathlib import Path

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"

# If you want to use wandb, 1) comment the following line
os.environ['WANDB_MODE'] = 'disabled'
# 2) ensure that there is a directory named wandb under the directory of this file
project_path = Path(__file__).parent.resolve()
os.environ['WANDB_DIR'] = f"{project_path}/wandb"
os.environ['WANDB_CACHE_DIR'] = f"{project_path}/wandb"
os.environ['WANDB_CONFIG_DIR'] = f"{project_path}/wandb"
os.environ['WANDB_DATA_DIR'] = f'{project_path}/wandb'
# 3) remember to fill your own repository information
# in the __init__ function of the WandbLinkLogger class
from utils.metrics import WandbLinkLogger
# then wandb will automatically track your experimental results

from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.TPNet import TPNet, RandomProjectionModule
from models.NAT import NAT
from models.modules import LinkPredictor_v1, LinkPredictor_v2
from utils.utils import set_thread, set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
import pickle as pk
import wandb

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"./logs/{args.prefix}_link_{args.dataset_name}_{args.model_name}.log", mode="w")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    # get data for training, validation and testing
    # PINT and TPNet require tuning of hyperparameters related to timestamps (i.e., args.pint_beta and args.time_decay_weight).
    # for continuous-time graphs, the time granularity is already in seconds.
    # to ensure consistent hyperparameter search range across all datasets, the time granularity of discrete-time graphs is also converted to seconds.
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                                 logger=logger, convert_time=(args.use_random_projection or args.model_name == 'PINT'))

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data,
                                                  sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids,
                                                 dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                                               seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids,
                                                        dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids,
                                                         dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))),
                                                batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))),
                                              batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))),
                                                       batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))),
                                               batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))),
                                                        batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):
        # set the seed to ensure the results can be reproduced. For reproducing NAT and TPNet, the deterministic_alg needs to be set to True
        set_random_seed(seed=run, deterministic_alg=args.use_random_projection or args.model_name == 'NAT')
        # set the maximum number of threads that can be utilized by the CPU.
        set_thread(3)

        args.seed = run

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        random_projections = None
        if args.use_random_projection:
            # create the model to maintain the temporal walk matrices
            random_projections = RandomProjectionModule(node_num=node_raw_features.shape[0],
                                                        edge_num=edge_raw_features.shape[0],
                                                        dim_factor=args.rp_dim_factor,
                                                        num_layer=args.rp_num_layer,
                                                        time_decay_weight=args.rp_time_decay_weight,
                                                        device=args.device, use_matrix=args.rp_use_matrix,
                                                        beginning_time=train_data.node_interact_times[0],
                                                        not_scale=args.rp_not_scale,
                                                        enforce_dim=args.enforce_dim)
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                    neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers,
                                    num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids,
                                                 train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                           neighbor_sampler=train_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name,
                                           num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift,
                                           src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                                           dst_node_std_time_shift=dst_node_std_time_shift, device=args.device,
                                           beta=args.pint_beta, num_hop=args.pint_hop)
        elif args.model_name == 'TPNet':
            dynamic_backbone = TPNet(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                     neighbor_sampler=train_neighbor_sampler,
                                     time_feat_dim=args.time_feat_dim,
                                     random_projections=None if args.encode_not_rp else random_projections,
                                     num_neighbors=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout,
                                     device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                    neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim,
                                    walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                   neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers,
                                   num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                          neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors,
                                          num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                         neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim,
                                         channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        elif args.model_name == 'NAT':
            dynamic_backbone = NAT(n_feat=node_raw_features, e_feat=edge_raw_features, time_dim=args.time_feat_dim,
                                   num_neighbors=[1] + args.nat_num_neighbors, dropout=args.dropout,
                                   n_hops=args.num_layers,
                                   ngh_dim=args.nat_ngh_dim, device=args.device)
            dynamic_backbone.set_seed(args.seed)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        # create the link prediction, NAT computes the edge embedding directly rather than node embedding, so it needs a different link predictor
        if args.model_name == 'NAT':
            link_predictor = LinkPredictor_v2(input_dim=node_raw_features.shape[1] + dynamic_backbone.self_dim * 2,
                                              hidden_dim=node_raw_features.shape[1] + dynamic_backbone.self_dim * 2,
                                              output_dim=1)
        else:
            link_predictor = LinkPredictor_v1(input_dim1=node_raw_features.shape[1],
                                              input_dim2=node_raw_features.shape[1],
                                              hidden_dim=node_raw_features.shape[1], output_dim=1,
                                              random_projections=None if args.decode_not_rp else random_projections,
                                              not_encode=args.not_encode)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate,
                                     weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_path = f"./saved_models/{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{args.seed}.pkl"
        early_stopping = EarlyStopping(patience=args.patience, save_model_path=save_model_path, logger=logger,
                                       model_name=args.model_name)

        loss_func = nn.BCEWithLogitsLoss()
        # wandb tracker, if you do not use wandb, ignore codes related to the wandb_logger
        wandb_logger = WandbLinkLogger('run', args)
        wandb_logger.watch(model)

        for epoch in range(args.num_epochs):

            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'TPNet', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'PINT']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()
            if args.model_name == 'NAT':
                # reinitialize ncache of NAT at the start of each epoch
                model[0].init_ncache()
            if args.use_random_projection:
                # reinitialize the random projections of temporal walk matrices at the start of each epoch
                random_projections.reset_random_projections()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                        train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids
                batch_neg_node_interact_times = batch_node_interact_times

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_neg_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                    # note that negative nodes do not change the memories while the positive nodes change the memories,
                    # we need to first compute the embeddings of negative nodes for memory-based models
                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_neg_node_interact_times,
                                                                          edge_ids=None,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          edges_are_positive=True,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['GraphMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_neg_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)
                elif args.model_name in ['DyGFormer', 'TPNet']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_neg_node_interact_times)
                elif args.model_name == 'NAT':
                    # get temporal embedding of negative links
                    # return one Tensor, with shape (batch_size, 3*node_feat_dim)
                    negative_edge_embeddings = \
                        model[0].compute_edge_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                  dst_node_ids=batch_neg_dst_node_ids,
                                                                  node_interact_times=batch_neg_node_interact_times,
                                                                  edge_ids=None,
                                                                  edges_are_positive=False)

                    # get temporal embedding of positive links, and update the ncache
                    # return one Tensor, with shape (batch_size, 3*node_feat_dim)
                    positive_edge_embeddings = \
                        model[0].compute_edge_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                  dst_node_ids=batch_dst_node_ids,
                                                                  node_interact_times=batch_node_interact_times,
                                                                  edge_ids=batch_edge_ids,
                                                                  edges_are_positive=True)
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                if args.model_name == 'NAT':
                    positive_probabilities = model[1](edge_embeddings=positive_edge_embeddings).squeeze(dim=-1)
                    negative_probabilities = model[1](edge_embeddings=negative_edge_embeddings).squeeze(dim=-1)
                else:
                    positive_probabilities = model[1](src_node_ids=batch_src_node_ids,
                                                      dst_node_ids=batch_dst_node_ids,
                                                      src_node_embeddings=batch_src_node_embeddings,
                                                      dst_node_embeddings=batch_dst_node_embeddings
                                                      ).squeeze(dim=-1)
                    negative_probabilities = model[1](src_node_ids=batch_neg_src_node_ids,
                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                      src_node_embeddings=batch_neg_src_node_embeddings,
                                                      dst_node_embeddings=batch_neg_dst_node_embeddings
                                                      ).squeeze(dim=-1)

                if args.use_random_projection:
                    # update the random projections of temporal walk matrices after observing positive links
                    random_projections.update(src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                                              node_interact_times=batch_node_interact_times)

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)],
                                   dim=0)
                loss = loss_func(input=predicts, target=labels)

                train_losses.append(loss.item())

                train_metrics.append(get_link_prediction_metrics(predicts=predicts.sigmoid(), labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(
                    f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model[0].memory_bank.detach_memory_bank()

            # backup memory bank/ncache/random projection after training so it can be used for new validation nodes
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
                if args.model_name == 'PINT':
                    train_backup_matrix_memory = model[0].matrix_memory.backup_memory()
            if args.model_name == 'NAT':
                train_backup_ncache = model[0].backup_ncache()
            if args.use_random_projection:
                train_backup_random_projections = random_projections.backup_random_projections()

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap,
                                                                     random_projections=random_projections)

            # backup memory bank/ncache/random projections after validating so it can be used for testing nodes
            # (since test edges are strictly later in time than validation edges)
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
                model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)
                if args.model_name == 'PINT':
                    val_backup_matrix_memory = model[0].matrix_memory.backup_memory()
                    model[0].matrix_memory.reload_memory(train_backup_matrix_memory)
            if args.model_name == 'NAT':
                val_backup_ncache = model[0].backup_ncache()
                model[0].reload_ncache(train_backup_ncache)
            if args.use_random_projection:
                val_backup_random_projections = random_projections.backup_random_projections()
                random_projections.reload_random_projections(train_backup_random_projections)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap,
                                                                                       random_projections=random_projections)

            # reload validation memory bank/ncache/random projections for testing nodes
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)
                if args.model_name == 'PINT':
                    model[0].matrix_memory.reload_memory(val_backup_matrix_memory)
            if args.model_name == 'NAT':
                model[0].reload_ncache(val_backup_ncache)
            if args.use_random_projection:
                random_projections.reload_random_projections(val_backup_random_projections)

            # perform testing once after test_interval_epochs
            test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                       model=model,
                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                       evaluate_idx_data_loader=test_idx_data_loader,
                                                                       evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                       evaluate_data=test_data,
                                                                       loss_func=loss_func,
                                                                       num_neighbors=args.num_neighbors,
                                                                       time_gap=args.time_gap,
                                                                       random_projections=random_projections)

            # reload validation memory bank/ncache/random projections for new testing nodes
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)
                if args.model_name == 'PINT':
                    model[0].matrix_memory.reload_memory(val_backup_matrix_memory)
            if args.model_name == 'NAT':
                model[0].reload_ncache(val_backup_ncache)
            if args.use_random_projection:
                random_projections.reload_random_projections(val_backup_random_projections)

            new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                         model=model,
                                                                                         neighbor_sampler=full_neighbor_sampler,
                                                                                         evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                         evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                         evaluate_data=new_node_test_data,
                                                                                         loss_func=loss_func,
                                                                                         num_neighbors=args.num_neighbors,
                                                                                         time_gap=args.time_gap,
                                                                                         random_projections=random_projections)

            # reload validation memory bank/ncache/random projections for saving models
            # note that since model treats memory/ncache/random projections as parameters, we need to reload the them for saving models
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)
                if args.model_name == 'PINT':
                    model[0].matrix_memory.reload_memory(val_backup_matrix_memory)
            if args.model_name == 'NAT':
                model[0].reload_ncache(val_backup_ncache)
            if args.use_random_projection:
                random_projections.reload_random_projections(val_backup_random_projections)

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(
                    f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(
                    f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(
                    f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')
            logger.info(f'test loss: {np.mean(test_losses):.4f}')
            for metric_name in test_metrics[0].keys():
                logger.info(
                    f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
            logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
            for metric_name in new_node_test_metrics[0].keys():
                logger.info(
                    f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')
            wandb_logger.log_epoch(train_losses=train_losses, train_metrics=train_metrics, val_losses=val_losses,
                                   val_metrics=val_metrics, test_losses=test_losses,
                                   test_metrics=test_metrics, new_node_val_losses=new_node_val_losses,
                                   new_node_val_metrics=new_node_val_metrics,
                                   new_node_test_losses=new_node_test_losses,
                                   new_node_test_metrics=new_node_test_metrics, epoch=epoch)

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append(
                    (metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model, args)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'---------get final performance on dataset {args.dataset_name}-------')

        # the saved best model of memory-based models cannot perform validation since the stored memory/ncache/random projections has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN', 'PINT', 'NAT'] and args.use_random_projection == False:
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap,
                                                                     random_projections=random_projections)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap,
                                                                                       random_projections=random_projections)

        # the memory/ncache/random projections in the best model have seen the validation edges, we need to backup the them for new testing nodes
        if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
            val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
            if args.model_name == 'PINT':
                val_backup_matrix_memory = model[0].matrix_memory.backup_memory()
        if args.model_name == 'NAT':
            val_backup_ncache = model[0].backup_ncache()
        if args.use_random_projection:
            val_backup_random_projections = random_projections.backup_random_projections()

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap,
                                                                   random_projections=random_projections)

        # reload validation memory bank/ncache/random projections for new testing nodes
        if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
            model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)
            if args.model_name == 'PINT':
                model[0].matrix_memory.reload_memory(val_backup_matrix_memory)
        if args.model_name == 'NAT':
            model[0].reload_ncache(val_backup_ncache)
        if args.use_random_projection:
            random_projections.reload_random_projections(val_backup_random_projections)

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                     model=model,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     loss_func=loss_func,
                                                                                     num_neighbors=args.num_neighbors,
                                                                                     time_gap=args.time_gap,
                                                                                     random_projections=random_projections)
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN', 'PINT', 'NAT'] and args.use_random_projection == False:
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                val_metric_dict[metric_name] = average_val_metric

            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                average_new_node_val_metric = np.mean(
                    [new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
                new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean(
                [new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')
        wandb_logger.log_run(test_losses=test_losses, test_metrics=test_metrics,
                             new_node_test_losses=new_node_test_losses,
                             new_node_test_metrics=new_node_test_metrics)
        wandb_logger.finish()

        if args.model_name not in ['JODIE', 'DyRep', 'TGN', 'PINT', 'NAT'] and args.use_random_projection == False:
            val_metric_all_runs.append(val_metric_dict)
            new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN', 'PINT', 'NAT'] and args.use_random_projection == False:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in
                                     val_metric_dict},
                "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for
                                              metric_name in new_node_val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in
                                 test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name
                                          in new_node_test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in
                                 test_metric_dict},
                "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name
                                          in new_node_test_metric_dict}
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_path = f"./saved_results/{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{args.seed}.json"

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the end of the log file
    if args.num_runs > 1:
        logger.info(f'-----------metrics over {args.num_runs} runs-----------')
        wandb_logger = WandbLinkLogger('summary', args)

        if args.model_name not in ['JODIE', 'DyRep', 'TGN', 'PINT', 'NAT'] and args.use_random_projection == False:
            for metric_name in val_metric_all_runs[0].keys():
                logger.info(
                    f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
                logger.info(
                    f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                    f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

            for metric_name in new_node_val_metric_all_runs[0].keys():
                logger.info(
                    f'new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}')
                logger.info(
                    f'average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}')

        for metric_name in test_metric_all_runs[0].keys():
            logger.info(
                f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
            logger.info(
                f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

        for metric_name in new_node_test_metric_all_runs[0].keys():
            logger.info(
                f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
            logger.info(
                f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')
        wandb_logger.log_final(run_metrics=test_metric_all_runs, new_node_run_metrics=new_node_test_metric_all_runs)

    sys.exit()
