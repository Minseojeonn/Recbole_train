# -*- coding: utf-8 -*-
# @Time   : 2022/3/24
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import os
from logging import getLogger
from time import time
import numpy as np
from tqdm import tqdm
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch
import torch.optim as optim

from recbole.data.interaction import Interaction
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator, Collector
from recbole.trainer import Trainer
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    EvaluatorType, get_tensorboard, set_color, get_gpu_usage, WandbLogger

DebiasTrainer = Trainer

class WRMFTrainer(DebiasTrainer):
    def __init__(self, config, model, train_data, dataset):
        super(WRMFTrainer, self).__init__(config, model)
        user_id = []
        item_id = []
        rating = []
        for i in train_data:
            user_id.append(i.user_id)
            item_id.append(i.item_id)
            rating.append(i.rating)
        user_id = torch.concat(user_id)
        item_id = torch.concat(item_id)
        rating = torch.concat(rating)
        self.rating_matrix = torch.sparse_coo_tensor(torch.stack([user_id, item_id]), rating, (dataset.n_users, dataset.n_items)).to(config['device'])
        #self.adj_matrix = torch.sparse_coo_tensor(torch.stack([user_id, item_id]), torch.ones_like(rating), (dataset.n_users, dataset.n_items))
        
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model(self.rating_matrix)
        total_loss = self.model.calculate_loss()
        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(
            valid_data, load_best_model=False, show_progress=show_progress
        )
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result
    
class DICETrainer(DebiasTrainer):
    r"""

    """

    def __init__(self, config, model):
        super(DICETrainer, self).__init__(config, model)
        self.decay = config['decay']

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

                Args:
                    train_data (DataLoader): the train data
                    valid_data (DataLoader, optional): the valid data, default: None.
                                                       If it's None, the early_stopping is invalid.
                    verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                    saved (bool, optional): whether to save the model parameters, default: True
                    show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                    callback_fn (callable): Optional callback function executed at end of epoch.
                                            Includes (epoch_idx, valid_score) input arguments.

                Returns:
                     (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
                """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = (sum(train_loss) if isinstance(train_loss, tuple) else train_loss)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics({"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx}, head="train", )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", "green")
                                      + " ["
                                      + set_color("time", "blue")
                                      + ": %.2fs, "
                                      + set_color("valid_score", "blue")
                                      + ": %f]"
                                      ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (set_color("valid result", "blue") + ": \n" + dict2str(valid_result))
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Valid_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, "valid_step": valid_step}, head="valid")

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                            epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

            if self.config['adaptive']:
                self.adapt_hyperparams()

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def adapt_hyperparams(self):
        self.model.adapt(self.decay)
        # self.sampler.adapt()
