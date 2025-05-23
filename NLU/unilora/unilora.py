"""
This file contains the implementation of UniLoRA and its derived classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np
import os
import sys


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "NLU"))
sys.path.append(BASE_DIR)
import config_device




class UniLoRA:
    def __init__(
        self, num_vectors, vector_length, in_features,out_features,rank
    ):
        """ """
        super().__init__()
        self.vector_bank = torch.nn.Parameter(torch.zeros(vector_length))
        self.register_buffer("project_params_A",torch.randint(0, vector_length, (in_features, rank), dtype=torch.long))
        self.register_buffer("project_params_B",torch.randint(0, vector_length,(rank, out_features), dtype=torch.long))
        self.register_buffer("norm_factor_A",torch.empty_like(self.project_params_A, dtype=torch.float))
        self.register_buffer("norm_factor_B",torch.empty_like(self.project_params_B, dtype=torch.float))

       

    # def _get_low_rank_matrix(self, logits):
    #     top_k_logits, indices = logits.topk(self.topk, dim=-1)
    #     topk_weights = F.softmax(top_k_logits, dim=-1)
    #     return (topk_weights.unsqueeze(-1) * self.vector_bank[indices]).sum(-2)


class UniLinear(nn.Linear, UniLoRA):
    def __init__(
        self,
        in_features,
        out_features,
        vector_length,
        num_vectors,
        rank,
        topk=2,
        fan_in_fan_out=False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        UniLoRA.__init__(
            self,
            num_vectors,
            vector_length,
            in_features = in_features,
            out_features = out_features,
            rank=rank,
        )

        self.fan_in_fan_out = fan_in_fan_out

    

        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
    # def init_weight(self):
    #     A = fastfood_torched(self.vector_bank, 1024*4, self.fastfood_params_A,config_device.training_args.device).view((1024,4))
    #     B = fastfood_torched(self.vector_bank, 1024*4, self.fastfood_params_B,config_device.training_args.device).view((4,1024))
    #     self.weight.data = self.weight.data - (A @ B).T

    def update_norm_factor(self, norm_factor_A,norm_factor_B):
        self.norm_factor_A = norm_factor_A
        self.norm_factor_B = norm_factor_B
   

    def forward(self, x):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        A = self.vector_bank[self.project_params_A] * self.norm_factor_A
        B = self.vector_bank[self.project_params_B] * self.norm_factor_B
        return F.linear(x, T(self.weight), bias=self.bias) + x @ A @ B
