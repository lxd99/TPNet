import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import wandb
import numpy as np
import torch.nn as nn


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


class WandbLinkLogger:
    def __init__(self, run_type, args):
        # If you want to use wandb, replace 'xx' with your own wandb project name and wandb entity name
        project_name, entity_name = 'xx', 'xx'
        if run_type == 'run':
            self.run_wandb = wandb.init(config=vars(args), project=project_name, entity=entity_name,
                                        tags=[run_type], reinit=True,
                                        name=f'{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{args.seed}')
        elif run_type == 'summary':
            self.run_wandb = wandb.init(config=vars(args), project=project_name, entity=entity_name,
                                        tags=[run_type], reinit=True,
                                        name=f'{args.prefix}_link_{args.dataset_name}_{args.model_name}_summary')
        elif run_type == 'eval_run':
            self.run_wandb = wandb.init(config=vars(args), project=project_name, entity=entity_name,
                                        tags=[run_type], reinit=True,
                                        name=f'{args.prefix}_link_{args.negative_sample_strategy}_{args.dataset_name}_{args.model_name}_seed{args.seed}')
        elif run_type == 'eval_summary':
            self.run_wandb = wandb.init(config=vars(args), project=project_name, entity=entity_name,
                                        tags=[run_type], reinit=True,
                                        name=f'{args.prefix}_link_{args.negative_sample_strategy}_{args.dataset_name}_{args.model_name}_summary')
        else:
            raise ValueError("Not Implemented Run Type")

    def watch(self, model: nn.Module):
        self.run_wandb.watch(model, log='gradients', log_freq=100)

    def standalize_results(self, results):
        metric_name_map = {'average_precision': 'AP', 'roc_auc': 'AUC'}
        task_prefix_map = {'transductive': '', 'inductive': 'i'}
        result_dict = {}
        for dtype in results:
            dtype_result_dict = {}
            for task_type in results[dtype]:
                task_prefix = task_prefix_map[task_type]
                data = results[dtype][task_type]
                for metric_name in data['metric'][0].keys():
                    task_metric_name = task_prefix + metric_name_map[metric_name]
                    average_metric = np.mean([metric[metric_name] for metric in data['metric']])
                    dtype_result_dict[task_metric_name] = np.around(average_metric, 4)
                dtype_result_dict[task_prefix + 'loss'] = np.around(np.mean(data['loss']), 4)
            result_dict[dtype] = dtype_result_dict
        return result_dict

    def log_epoch(self, train_losses, train_metrics, val_losses, val_metrics, test_losses, test_metrics,
                  new_node_val_losses, new_node_val_metrics, new_node_test_losses, new_node_test_metrics,
                  epoch):
        results = {
            'train': {
                'transductive': {
                    'loss': train_losses,
                    'metric': train_metrics
                }
            },
            'val': {
                'transductive': {
                    'loss': val_losses,
                    'metric': val_metrics
                },
                'inductive': {
                    'loss': new_node_val_losses,
                    'metric': new_node_val_metrics
                }
            },
            'test': {
                'transductive': {
                    'loss': test_losses,
                    'metric': test_metrics
                },
                'inductive': {
                    'loss': new_node_test_losses,
                    'metric': new_node_test_metrics
                }
            }
        }
        result_dict = self.standalize_results(results)
        self.run_wandb.log(result_dict, step=epoch)

    def log_run(self, test_losses, test_metrics, new_node_test_losses, new_node_test_metrics):
        results = {
            'test': {
                'transductive': {
                    'loss': test_losses,
                    'metric': test_metrics
                },
                'inductive': {
                    'loss': new_node_test_losses,
                    'metric': new_node_test_metrics
                }
            }
        }
        result_dict = self.standalize_results(results)['test']
        self.run_wandb.summary.update(result_dict)

    def log_final(self, run_metrics, new_node_run_metrics):
        def final_standalize_results(results):
            metric_name_map = {'average_precision': 'AP', 'roc_auc': 'AUC'}
            task_prefix_map = {'transductive': '', 'inductive': 'i'}
            result_dict_mean = {}
            result_dict_std = {}
            for task_type in results:
                task_prefix = task_prefix_map[task_type]
                data = results[task_type]
                for metric_name in data[0].keys():
                    task_metric_name = task_prefix + metric_name_map[metric_name]
                    average_metric = np.mean([metric[metric_name] for metric in data])
                    std_metric = np.std([metric[metric_name] for metric in data], ddof=1)
                    result_dict_mean[task_metric_name] = np.around(average_metric, 4)
                    result_dict_std[task_metric_name + '_std'] = np.around(std_metric, 4)
            result_dict_mean.update(result_dict_std)
            return result_dict_mean

        results = {
            'transductive': run_metrics,
            'inductive': new_node_run_metrics
        }
        result_dict = final_standalize_results(results)
        self.run_wandb.summary.update(result_dict)

    def finish(self):
        self.run_wandb.finish()
