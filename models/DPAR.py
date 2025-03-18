import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import numpy as np
import scipy as sp

class DPAR(nn.Module):

    def __init__ (self, params, device):
        super().__init__()
        self.W1 = torch.tensor(params['W1:0']).to(device)
        self.W2 = torch.tensor(params['W2:0']).to(device)
        self.nclass = 7
        self.nfeat = 2879
        self.hidden_sizes = 32
        self.ReLU = nn.ReLU()
        self.device = device

    def forward(self,input,adj):

        input = input.to(self.device)
        adj = adj.to(self.device)

        if input.data.is_sparse:
            hidden_1 = torch.spmm(input, (self.W1))
        else:
            hidden_1 = torch.mm(input, (self.W1))

        hidden_1 = self.ReLU(hidden_1)

        local_logits = torch.mm(hidden_1, (self.W2))
        # logits = self.ReLU(local_logits) ##<=======
        logits = local_logits

        alpha = 0.25
        deg_row = adj.sum(1)

        deg_row_inv_alpha = (1 - alpha) / torch.clamp(deg_row, min=1e-12)
        for i in range(2):
            logits = deg_row_inv_alpha[:, None] * (adj @ logits) + alpha * local_logits

        return F.log_softmax(logits, dim=1)

