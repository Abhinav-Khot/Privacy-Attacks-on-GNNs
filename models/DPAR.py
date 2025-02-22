import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import numpy as np
import scipy as sp

class DPAR(nn.Module):

    def __init__ (self, params):
        self.W1 = params['W1:0']
        self.W2 = params['W2:0']

    def forward(self,input,adj):

        if input.data.is_sparse:
            hidden_1 = torch.spmm(input, self.W1)
        else:
            hidden_1 = torch.mm(input, self.W1)
        
        hidden_1 = nn.relu(hidden_1)

        local_logits = torch.mm(hidden_1,self.W2)
        logits = local_logits

        alpha = 0.25
        deg_row = adj.sum(1).A1
        deg_row_inv_alpha = (1 - alpha) / np.maximum(deg_row, 1e-12)
        for i in range(2):
            logits = deg_row_inv_alpha[:, None] * (adj @ logits) + alpha * local_logits

        return F.log_softmax(logits, dim=1)
        
        