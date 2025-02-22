import torch 
import ppr_utils as ppr
import time
import numpy as np
import scipy as sp

class DPAR(nn.Module):

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
