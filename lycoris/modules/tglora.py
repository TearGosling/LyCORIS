import math
from weakref import ref

import torch
import torch.nn as nn
import torch.nn.functional as F


class TGLoRAModule(nn.Module):
    """
    modifed from kohya-ss/sd-scripts/networks/lora:LoRAModule
    """

    def __init__(
        self, 
        lora_name, org_module: nn.Module, 
        multiplier=1.0, 
        lora_dim=4, alpha=1, 
        dropout=0., rank_dropout=0., module_dropout=0.,
        *args,
        **kwargs,
    ):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.cp = False

        if isinstance(org_module, nn.Conv2d):
            assert org_module.kernel_size == (1,1)
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.ad = nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
            self.au = nn.Conv2d(lora_dim, in_dim, (1, 1), bias=False)
            self.bd = nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
            self.bu = nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
        elif isinstance(org_module, nn.Linear):
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.ad = nn.Linear(in_dim, lora_dim, bias=False)
            self.au = nn.Linear(lora_dim, in_dim, bias=False)
            self.bd = nn.Linear(in_dim, lora_dim, bias=False)
            self.bu = nn.Linear(lora_dim, out_dim, bias=False)
        else:
            raise NotImplementedError
        self.shape = org_module.weight.shape
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        
        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha)) # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.ad.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.au.weight)
        torch.nn.init.kaiming_uniform_(self.bd.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.bu.weight)

        self.multiplier = multiplier
        self.org_module = [org_module]

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def make_weight(self, device=None):
        wad = self.ad.weight.view(self.ad.weight.size(0), -1)
        wau = self.au.weight.view(self.au.weight.size(0), -1)
        wbd = self.bd.weight.view(self.bd.weight.size(0), -1)
        wbu = self.bu.weight.view(self.bu.weight.size(0), -1)
        orig = self.org_module[0].weight.view(self.org_module[0].weight.size(0), -1)
        return ((wbu @ wbd) + ((orig @ wau) @ wad))

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.scale * self.multiplier
        
        ax_mid = F.silu(self.ad(x)) * scale
        bx_mid = F.silu(self.bd(x)) * scale
        
        if self.rank_dropout and self.training:
            drop_a = torch.rand(self.lora_dim, device=ax_mid.device) < self.rank_dropout
            drop_b = torch.rand(self.lora_dim, device=bx_mid.device) < self.rank_dropout
            if (dims:=len(x.shape)) == 4:
                drop_a = drop_a.view(1, -1, 1, 1)
                drop_b = drop_b.view(1, -1, 1, 1)
            else:
                drop_a = drop_a.view(*[1]*(dims-1), -1)
                drop_b = drop_b.view(*[1]*(dims-1), -1)
            ax_mid = ax_mid * drop_a
            bx_mid = bx_mid * drop_b
        
        return self.org_forward(x + self.dropout(self.au(ax_mid))) + self.dropout(self.bu(bx_mid))
