import torch
import torch.jit
import logging
from typing import List, Dict

import time
import torch.nn.functional as F
import torch.nn as nn


def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):

    E_list = []
    oldE = float('inf')
    Y = (-unary).softmax(-1)  # [N, K]
    for i in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
        exponent = -unary + pairwise
        Y = exponent.softmax(-1)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)

        if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
            # logger.info(f'Converged in {i} iterations')
            # print('Converged in iterations:', i)
            break
        else:
            oldE = E

    return Y


def entropy_energy(Y, unary, pairwise, bound_lambda):
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E


class AffinityMatrix:

    def __init__(self, **kwargs):
        pass

    def __call__(X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]
        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W


class LAME(nn.Module):
    def __init__(self, knn=5):
        self.knn = knn
        self.sigma = 1.0  # from overall_best.yaml in LAME github
        self.affinity = kNN_affinity(knn=self.knn)

    # def format_result(self, batched_inputs, probas):
    #     with torch.no_grad():
    #         pred_classes = probas.argmax(-1, keepdim=True)
    #         scores = probas.max(-1, keepdim=True)

    def __call__(self, scores_raw, feats):
        ### 1.0
        _,_,H,W = scores_raw.size()
        unary = - torch.log(scores_raw.reshape(-1, H*W) + 1e-10)  # [N, K]  batch 324
        feats2 = F.normalize(feats.flatten(-3).squeeze(0), p=2, dim=-1)  # [N, d]  [1 324*768]
        kernel = self.affinity(feats2)  # [N, N]  batch batch
        # kernel = 1/2 * (kernel + kernel.t())
        # --- Perform optim ---
        Y = laplacian_optimization(unary, kernel)  # [N, K]  new score
        return Y

        ### 2.0 new one scores_raw 324 2
        # neg_score = torch.ones_like(scores_raw) - scores_raw
        # new_score = torch.cat([scores_raw,neg_score], dim=1)    # 1 2 18 18
        # unary = - torch.log(new_score.flatten(-2).squeeze(0).transpose(1, 0) + 1e-10)  # [N, K]  324 2
        # feats2 = F.normalize(feats.flatten(-3).squeeze(0).transpose(1, 0), p=2, dim=-1)  # [N, d]  324 768
        # kernel = self.affinity(feats2)  # [N, N]  batch batch  324 324
        # Y = laplacian_optimization(unary, kernel)  # [N, K]  new score 324 2
        # return Y[:, :1].transpose(1, 0)   # 1 324

