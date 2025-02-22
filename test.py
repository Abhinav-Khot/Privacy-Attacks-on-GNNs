import numpy as np

b = np.load('matricx.npz', allow_pickle = True)
print(b)
print(b['train_adj_matrix'])