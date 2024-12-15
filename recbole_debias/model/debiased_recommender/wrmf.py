# -*- coding: utf-8 -*-
# @Time   : 2022/4/20
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

r"""
MACR
################################################
Reference:
    Tianxin Wei et al, "Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System"
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.model.layers import MLPLayers
from recbole.utils import InputType
import torch.nn.init as init
from recbole_debias.model.abstract_recommender import DebiasedRecommender


class WRMF(DebiasedRecommender):
    r"""
        WRMF model
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(WRMF, self).__init__(config, dataset)
        self.LABEL = config['LABEL_FIELD']
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.item_weight = config['alpha']
        self.reg_weight = config['lambda']
        self.device = config['device']
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        init.xavier_normal_(self.user_embedding.weight)
        init.xavier_normal_(self.item_embedding.weight)
        # matrix
        # dummy loss
        self.dummy_loss = torch.tensor(0.0, requires_grad=True)

        # parameters initialization
        


    def compute_xu(self, Y, Wu, Au, lambda_):
        """
        Update user factors using weighted regularized least squares.
        Y: item matrix
        Wu: diagonal weight matrix for a specific user
        Au: interaction vector for a specific user
        lambda_: regularization parameter
        """
        # Compute weighted least squares: X_u = (Y^T W Y + lambda * I)^-1 (Y^T W A_u)
        W = torch.diag(Wu).to(self.device)  # Ensure Wu is diagonal
        regularization = lambda_ * torch.eye(Y.size(1), device=Y.device, dtype=Y.dtype).to(self.device)
        YtWY = Y.T @ W @ Y
        YtWAu = Y.T @ W @ Au
        Xu = torch.linalg.solve(YtWY + regularization, YtWAu).to(self.device)
        return Xu

    def compute_yi(self, X, Wi, Ai, lambda_):
        """
        Update item factors using weighted regularized least squares.
        X: user matrix
        Wi: diagonal weight matrix for a specific item
        Ai: interaction vector for a specific item
        lambda_: regularization parameter
        """
        W = torch.diag(Wi)  # Ensure Wi is diagonal
        regularization = lambda_ * torch.eye(X.size(1), device=X.device, dtype=X.dtype).to(self.device)
        XtWX = X.T @ W @ X
        XtWAi = X.T @ W @ Ai
        Yi = torch.linalg.solve(XtWX + regularization, XtWAi).to(self.device)
        return Yi

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, rating_matrix):
        with torch.no_grad():
            # Access embedding weights
            user_e = self.user_embedding.weight  # User embedding weights
            item_e = self.item_embedding.weight  # Item embedding weights

            # Update user factors directly in place
            for i, Ri in enumerate(rating_matrix):
                Wi = 1 + self.item_weight * Ri.to_dense()  # Define weight vector for user i       
                user_e[i] = self.compute_xu(item_e, Wi, Ri, self.reg_weight)  # Update user embedding in place

            # Update item factors directly in place
            for j, Rj in enumerate(rating_matrix.T):
                Wj = 1 + self.item_weight * Rj.to_dense()  # Define weight vector for item j
                item_e[j] = self.compute_yi(user_e, Wj, Rj, self.reg_weight)  # Update item embedding in place
            
        return user_e, item_e

    def calculate_loss(self):
        return self.dummy_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        score = torch.sum(user_e * item_e, dim=1)
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.T)
        return score
