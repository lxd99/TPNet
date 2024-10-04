import argparse
import sys
import torch


def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--prefix', type=str, default='test', help='name of the experiment')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci',
                                 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGFormer',
                        help='name of the model, note that EdgeBank is only applicable for evaluation',
                        choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'EdgeBank', 'TCL', 'GraphMixer', 'DyGFormer',
                                 'TimeTop', 'TPNet', 'PINT', 'NAT'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent',
                        choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float,
                        help='the hyperparameter that controls the sampling preference with time interval, '
                             'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                             'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8,
                        help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory',
                        help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion',
                        help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32,
                        help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'],
                        help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--negative_sample_strategy', type=str, default='random',
                        choices=['random', 'historical', 'inductive', 'new_random'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--load_best_configs', action='store_true', default=False,
                        help='whether to load the best configurations')
    parser.add_argument('--pint_beta', type=float, default=0.1, help='the beat hyperparameter of PINT')
    parser.add_argument('--pint_hop', type=int, default=3, help='the hop of the PINT walk matrix')
    parser.add_argument('--nat_ngh_dim', type=int, default=4, help='the dimension of NAT ncahche')
    parser.add_argument('--nat_num_neighbors', type=int, nargs='*', default=[32, 16],
                        help='a list of neighbor sampling numbers for different hops of NAT')
    parser.add_argument('--use_random_projection', action='store_true', help='whether use the random projection')
    parser.add_argument('--rp_num_layer', type=int, default=3,
                        help='the max hop of the maintained temporal walk matrices')
    parser.add_argument('--rp_time_decay_weight', type=float, default=0.0000001,
                        help='the time decay weight of the maintained temporal walk matrix')
    parser.add_argument('--rp_dim_factor', type=int, default=10,
                        help='the dim factor of random projections w.r.t. the log(2*edge_num)')
    parser.add_argument('--encode_not_rp', action='store_true', help='whether to use pairwise features in encoder')
    parser.add_argument('--decode_not_rp', action='store_true', help='whether to use pairwise features in link decoder')
    parser.add_argument('--rp_not_scale', action='store_true',
                        help='whether to scale and relu for inner product of random projections')
    parser.add_argument('--not_encode', action='store_true',
                        help='whether to use encoder to learn node embeddings from historical interactions')
    parser.add_argument('--enforce_dim', type=int, default=-1,
                        help='whether to specific the dimension of random projections')
    parser.add_argument('--rp_use_matrix', action='store_true',
                        help='whether to explicitly maintain temporal walk matrices')
    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    if args.model_name == 'EdgeBank':
        assert is_evaluation, 'EdgeBank is only applicable for evaluation!'

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name in ['TGAT']:
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ['enron', 'CanParl', 'UNvote']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        if args.dataset_name in ['reddit', 'CanParl', 'UNtrade']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        args.num_neighbors = 10
        args.num_layers = 1
        if args.model_name == 'JODIE':
            if args.dataset_name in ['mooc', 'USLegis']:
                args.dropout = 0.2
            elif args.dataset_name in ['lastfm']:
                args.dropout = 0.3
            elif args.dataset_name in ['uci', 'UNtrade']:
                args.dropout = 0.4
            elif args.dataset_name in ['CanParl']:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        elif args.model_name == 'DyRep':
            if args.dataset_name in ['mooc', 'lastfm', 'enron', 'uci', 'CanParl', 'USLegis', 'Contacts']:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        else:
            assert args.model_name == 'TGN'
            if args.dataset_name in ['mooc', 'UNtrade']:
                args.dropout = 0.2
            elif args.dataset_name in ['lastfm', 'CanParl']:
                args.dropout = 0.3
            elif args.dataset_name in ['enron', 'SocialEvo']:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        if args.model_name in ['TGN', 'DyRep']:
            if args.dataset_name in ['CanParl'] or (args.model_name == 'TGN' and args.dataset_name == 'UNvote'):
                args.sample_neighbor_strategy = 'uniform'
            else:
                args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'CAWN':
        args.time_scaling_factor = 1e-6
        if args.dataset_name in ['mooc', 'SocialEvo', 'uci', 'Flights', 'UNtrade', 'UNvote', 'Contacts']:
            args.num_neighbors = 64
        elif args.dataset_name in ['lastfm', 'CanParl']:
            args.num_neighbors = 128
        else:
            args.num_neighbors = 32
        if args.dataset_name in ['CanParl']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
        args.sample_neighbor_strategy = 'time_interval_aware'
    elif args.model_name == 'EdgeBank':
        if args.negative_sample_strategy == 'random':
            if args.dataset_name in ['wikipedia', 'reddit', 'uci', 'Flights']:
                args.edge_bank_memory_mode = 'unlimited_memory'
            elif args.dataset_name in ['mooc', 'lastfm', 'enron', 'CanParl', 'USLegis']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'fixed_proportion'
            elif args.dataset_name in ['UNtrade', 'UNvote', 'Contacts']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'repeat_interval'
            else:
                assert args.dataset_name == 'SocialEvo'
                args.edge_bank_memory_mode = 'repeat_threshold_memory'
        elif args.negative_sample_strategy == 'historical':
            if args.dataset_name in ['uci', 'CanParl', 'USLegis']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'fixed_proportion'
            elif args.dataset_name in ['mooc', 'lastfm', 'enron', 'UNtrade', 'UNvote', 'Contacts']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'repeat_interval'
            else:
                assert args.dataset_name in ['wikipedia', 'reddit', 'SocialEvo', 'Flights']
                args.edge_bank_memory_mode = 'repeat_threshold_memory'
        else:
            assert args.negative_sample_strategy == 'inductive'
            if args.dataset_name in ['USLegis']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'fixed_proportion'
            elif args.dataset_name in ['uci', 'UNvote']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'repeat_interval'
            else:
                assert args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron',
                                             'SocialEvo', 'Flights', 'CanParl', 'UNtrade', 'Contacts']
                args.edge_bank_memory_mode = 'repeat_threshold_memory'
    elif args.model_name == 'TCL':
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ['SocialEvo', 'uci', 'UNtrade', 'UNvote', 'Contacts']:
            args.dropout = 0.0
        elif args.dataset_name in ['CanParl']:
            args.dropout = 0.2
        elif args.dataset_name in ['USLegis']:
            args.dropout = 0.3
        else:
            args.dropout = 0.1
        if args.dataset_name in ['reddit', 'CanParl', 'USLegis', 'UNtrade', 'UNvote']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'GraphMixer':
        args.num_layers = 2
        if args.dataset_name in ['wikipedia']:
            args.num_neighbors = 30
        elif args.dataset_name in ['reddit', 'lastfm']:
            args.num_neighbors = 10
        else:
            args.num_neighbors = 20
        if args.dataset_name in ['wikipedia', 'reddit', 'enron']:
            args.dropout = 0.5
        elif args.dataset_name in ['mooc', 'uci', 'USLegis']:
            args.dropout = 0.4
        elif args.dataset_name in ['lastfm', 'UNvote']:
            args.dropout = 0.0
        elif args.dataset_name in ['SocialEvo']:
            args.dropout = 0.3
        elif args.dataset_name in ['Flights', 'CanParl']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        if args.dataset_name in ['CanParl', 'UNtrade', 'UNvote']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name in ['DyGFormer']:
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        elif args.dataset_name in ['mooc', 'enron', 'Flights', 'USLegis', 'UNtrade']:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif args.dataset_name in ['lastfm']:
            args.max_input_sequence_length = 512
            args.patch_size = 16
        elif args.dataset_name in ['CanParl']:
            args.max_input_sequence_length = 2048
            args.patch_size = 64
        elif args.dataset_name in ['UNvote']:
            args.max_input_sequence_length = 128
            args.patch_size = 4
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit', 'UNvote']:
            args.dropout = 0.2
        elif args.dataset_name in ['enron', 'USLegis', 'UNtrade', 'Contacts']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
    elif args.model_name == 'TPNet':
        if args.dataset_name == 'CanParl':
            args.sample_neighbor_strategy = 'uniform'
    elif args.model_name == 'PINT':
        # number of layers
        if args.dataset_name in ['UNvote']:
            args.num_layers = 1
        else:
            args.num_layers = 2
        # beta decay weight
        if args.dataset_name == 'mooc':
            args.num_neighbors = 10
        else:
            args.num_neighbors = 20
        # beta
        if args.dataset_name in ['wikipedia', 'lastfm', 'mooc', 'SocialEvo', 'Contacts']:
            args.pint_beta = 0.0001
        elif args.dataset_name in ['uci', 'reddit', 'enron']:
            args.pint_beta = 0.00001
        elif args.dataset_name in ['CanParl', 'UNtrade', 'Flights']:
            args.pint_beta = 0.000001
        elif args.dataset_name in ['USLegis', 'UNvote']:
            args.pint_beta = 0.0000001
        else:
            raise ValueError("Not Recognized Dataset!")
    elif args.model_name == 'NAT':
        if args.dataset_name in ['wikipedia', 'CanParl', 'SocialEvo', 'mooc', 'lastfm', 'UNvote']:
            args.nat_ngh_dim = 4
        elif args.dataset_name in ['uci', 'enron', 'UNtrade']:
            args.nat_ngh_dim = 8
        elif args.dataset_name in ['USLegis', 'reddit', 'Flights', 'Contacts']:
            args.nat_ngh_dim = 16
        else:
            raise ValueError("Not Recognized Dataset!")

        if args.dataset_name in ['uci', 'CanParl', 'UNtrade']:
            args.nat_num_neighbors = [32, 16]
        elif args.dataset_name in ['reddit', 'wikipedia', 'enron', 'mooc', 'USLegis', 'UNvote', 'Contacts', 'SocialEvo',
                                   'Flights', 'lastfm']:
            args.nat_num_neighbors = [64, 32]
        else:
            raise ValueError("Not Recognized Dataset!")
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")

    if args.use_random_projection:
        if args.dataset_name in ['Contacts']:
            args.rp_time_decay_weight = 0.0001
        elif args.dataset_name in ['UNvote', 'CanParl', 'mooc']:
            args.rp_time_decay_weight = 0.00001
        elif args.dataset_name in ['wikipedia', 'reddit', 'enron', 'SocialEvo', 'Flights', 'USLegis']:
            args.rp_time_decay_weight = 0.000001
        elif args.dataset_name in ['lastfm', 'uci', 'UNtrade']:
            args.rp_time_decay_weight = 0.0000001
        else:
            raise ValueError("Not Recognized Dataset!")
