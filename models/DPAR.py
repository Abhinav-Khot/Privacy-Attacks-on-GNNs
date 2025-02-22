import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ppr_utils as ppr
import time
import numpy as np
import scipy as sp

class DPAR(nn.Module):

    def __init__(self, train_adj_matrix, train_index, ppr_num, alpha, eps, rho, topk, sigma_ista, clip_bound_ista, dp_ppr, em_sensitivity, report_val_eps, EM, EM_eps):
        super(DPAR, self).__init__()
        self.train_adj_matrix = train_adj_matrix
        self.train_index = train_index
        self.ppr_num = ppr_num
        self.alpha = alpha
        self.eps = eps
        self.rho = rho
        self.topk = topk
        self.sigma_ista = sigma_ista
        self.clip_bound_ista = clip_bound_ista
        self.dp_ppr = dp_ppr
        self.em_sensitivity = em_sensitivity
        self.report_val_eps = report_val_eps
        self.EM = EM
        self.EM_eps = EM_eps


    def calculate_ppr_matrix(train_adj_matrix, train_index, ppr_num, alpha, eps, rho, topk, sigma_ista, clip_bound_ista, dp_ppr, em_sensitivity, report_val_eps, EM, EM_eps):
        start = time.time()
        topk_train = ppr.topk_ppr_matrix_ista(train_adj_matrix, alpha, eps, rho, train_index[:ppr_num],
                                                topk, sigma_ista, clip_bound_ista, dp_ppr,
                                                em_sensitivity, report_val_eps, EM, EM_eps)
        if ppr_num < len(train_index):
            topk_train_I = np.identity(len(train_index))
            topk_train_I = topk_train_I[ppr_num:]
            topk_train_dense = topk_train.toarray()
            topk_train_dense_full = np.concatenate((topk_train_dense, topk_train_I), axis=0)
            topk_train = sp.csr_matrix(topk_train_dense_full)

        time_preprocessing = time.time() - start
        print(f"Calculate Train PPR Matrix Runtime: {time_preprocessing:.2f}s")

        # normalize l1 norm of each column of topk_train'''
        topk_train_dense = topk_train.toarray()
        for col in range(len(topk_train_dense[0, :])):
            if np.linalg.norm(topk_train_dense[:, col], ord=1) != 0:
                topk_train_dense[:, col] *= (1.0 / np.linalg.norm(topk_train_dense[:, col], ord=1))
        topk_train = sp.csr_matrix(topk_train_dense)

        print(topk_train)
        return topk_train


if __name__ == __main__:
    train_adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    train_index = np.array([0, 1, 2])
    ppr_num = 2
    alpha = 0.1
    eps = 0.1
    rho = 0.1
    topk = 2
    sigma_ista = 0.1
    clip_bound_ista = 0.1
    dp_ppr = 0.1
    em_sensitivity = 0.1
    report_val_eps = 0.1
    EM = 0.1
    EM_eps = 0.1

    dpar = DPAR(train_adj_matrix, train_index, ppr_num, alpha, eps, rho, topk, sigma_ista, clip_bound_ista, dp_ppr, em_sensitivity, report_val_eps, EM, EM_eps)
    dpar.calculate_ppr_matrix(train_adj_matrix, train_index, ppr_num, alpha, eps, rho, topk, sigma_ista, clip_bound_ista, dp_ppr, em_sensitivity, report_val_eps, EM, EM_eps)