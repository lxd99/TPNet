import os
import torch
import torch.nn as nn
import logging
import argparse


class EarlyStopping(object):

    def __init__(self, patience: int, save_model_path: str, logger: logging.Logger,
                 model_name: str = None):
        """
        Early stop strategy.
        :param patience: int, max patience
        :param save_model_path: str, save model folder
        :param logger: Logger
        :param model_name: str, model name
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.logger = logger
        self.save_model_path = save_model_path
        self.model_name = model_name

    def step(self, metrics: list, model: nn.Module, args: argparse.Namespace):
        """
        execute the early stop strategy for each evaluation process
        :param metrics: list, list of metrics, each element is a tuple (str, float, boolean) -> (metric_name, metric_value, whether higher means better)
        :param model: nn.Module
        :param args: argparse.Namespace
        :return:
        """
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.save_checkpoint(model, args)
            self.counter = 0
        # metrics are not better at the epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: nn.Module, args: argparse.Namespace):
        """
        :param model: nn.Module
        :param args:
        :return:
        """
        self.logger.info(f"save model {self.save_model_path}")
        save_object = {'model': model.state_dict(), 'args': vars(args)}
        if self.model_name in ['JODIE', 'DyRep', 'TGN','PINT']:
            save_object['message'] = model[0].memory_bank.node_raw_messages
        torch.save(save_object, self.save_model_path)

    def load_checkpoint(self, model: nn.Module, map_location: str = None):
        """
        load model at self.save_model_path
        :param model: nn.Module
        :param map_location: str, how to remap the storage locations
        :return:
        """
        self.logger.info(f"load model {self.save_model_path}")
        save_object = torch.load(self.save_model_path, map_location=map_location)
        model.load_state_dict(save_object['model'])
        if self.model_name in ['JODIE', 'DyRep', 'TGN','PINT']:
            model[0].memory_bank.node_raw_messages = save_object['message']
