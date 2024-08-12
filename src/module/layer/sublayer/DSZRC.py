import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import math

from .ImportantScore import ImportantScore

class SoftChooseDenseGrad(Function):
    @staticmethod
    def forward(ctx, imp_A, imp_B, q_top, q_bot, temperature, **kwargs):
        # imp_A: (b, h, n_top, 1)
        # imp_B: (b, h, n_bot, 1)
        # q_top: (b, h, n_top, d)
        # q_bot: (b, h, n_bot, d)
        # ret: (b, h, n_top, d)
        # soft

        imp_B = imp_B.transpose(2, 3)
        imp_AB = imp_A - imp_B
        p = torch.sigmoid(imp_AB / temperature)  # (b, h, n_top, n_bot)
        _1_p = 1 - p
        p_mean_3 = p.mean(dim=3, keepdim=True)
        
        ctx.save_for_backward(imp_A, imp_B, q_top, q_bot, temperature)  # Save for backward
        return p_mean_3 * q_top + _1_p / p.shape[3] @ q_bot


    @staticmethod
    def backward(ctx, grad_output):
        imp_A, imp_B, q_top, q_bot, temperature = ctx.saved_tensors

        imp_AB = imp_A - imp_B
        p = torch.sigmoid(imp_AB / temperature)  # (b, h, n_top, n_bot)
        _1_p = 1 - p
        p_mean_3 = p.mean(dim=3, keepdim=True)
        d_a = 1
        d_b = -1

        grad_imp_A = grad_imp_B = grad_q_top = grad_q_bot = None

        sigmoid_grad = p * (1 - p) / temperature  # derivative of sigmoid
        max_sigmoid_grad = torch.amax(sigmoid_grad, dim=(0, 2, 3), keepdim=True)[0]
        sigmoid_grad = sigmoid_grad * torch.clamp(max_sigmoid_grad, 0.0, 0.25) / max_sigmoid_grad

        term1 = (q_top * grad_output).sum(dim=3, keepdim=True) * sigmoid_grad
        term2 = (grad_output @ q_bot.transpose(2, 3) * sigmoid_grad) * -1 / p.shape[3]

        grad_imp_A = (term1 * d_a).mean(dim=3, keepdim=True) + (term2 * d_a).sum(dim=3, keepdim=True)
        grad_imp_B = ((term1 * d_b).sum(dim=2, keepdim=True) / term1.shape[3] + (term2 * d_b).sum(dim=2, keepdim=True)).transpose(2, 3)

        grad_q_top = grad_output * p_mean_3  # (b, h, n_top, d)

        grad_q_bot = _1_p.transpose(2, 3) @ grad_output / p.shape[3] # (b, h, n_bot, d)

        return grad_imp_A, grad_imp_B, grad_q_top, grad_q_bot, None

class HardChooseDenseGrad(Function):
    @staticmethod
    def forward(ctx, imp_A, imp_B, q_top, q_bot, temperature, mean_grad_output):
        # imp_A: (b, h, n_top, 1)
        # imp_B: (b, h, n_bot, 1)
        # q_top: (b, h, n_top, d)
        # q_bot: (b, h, n_bot, d)
        # ret: (b, h, n_top, d)
        # hard
        imp_B = imp_B.transpose(2, 3)
        ctx.save_for_backward(imp_A, imp_B, q_top, q_bot, temperature, mean_grad_output)  # Save for backward
        return q_top


    @staticmethod
    def backward(ctx, grad_output):
        imp_A, imp_B, q_top, q_bot, temperature, mean_grad_output = ctx.saved_tensors

        imp_AB = imp_A - imp_B
        p = torch.sigmoid(imp_AB / temperature)  # (b, h, n_top, n_bot)
        _1_p = 1 - p
        p_mean_3 = p.mean(dim=3, keepdim=True)
        d_a = 1
        d_b = -1

        grad_imp_A = grad_imp_B = grad_q_top = None

        sigmoid_grad = p * (1 - p) / temperature  # derivative of sigmoid
        max_sigmoid_grad = torch.amax(sigmoid_grad, dim=(0, 2, 3), keepdim=True)[0]
        sigmoid_grad = sigmoid_grad * torch.clamp(max_sigmoid_grad, 0.0, 0.25) / max_sigmoid_grad

        term1 = (q_top * grad_output).sum(dim=3, keepdim=True) * sigmoid_grad
        term2 = (grad_output @ q_bot.transpose(2, 3) * sigmoid_grad) * -1 / p.shape[3]

        grad_imp_A = (term1 * d_a).mean(dim=3, keepdim=True) + (term2 * d_a).sum(dim=3, keepdim=True)
        grad_imp_B = ((term1 * d_b).sum(dim=2, keepdim=True) / term1.shape[3] + (term2 * d_b).sum(dim=2, keepdim=True)).transpose(2, 3)

        grad_q_top = grad_output # (b, h, n_top, d)

        return grad_imp_A, grad_imp_B, grad_q_top, None, None, 2 * (mean_grad_output - grad_output)

def get_gumbel(tensor):
    return tensor - torch.log(-torch.log(torch.rand_like(tensor) + 1e-20) + 1e-20)

class DSZRC(nn.Module):
    def __init__(self,
                 nhead: int, 
                 n_sampled_token: int,
                 n_dimension_x: int,
                 n_dimension_qk: int,
                 temperature,
                 hard):
        super().__init__()
        self.nhead = nhead
        self.n_sampled_token = n_sampled_token
        # Define a linear transformation to compute the importance score for each token
        # This layer maps from n_dimension to nhead
        self.importance_score = ImportantScore(nhead, n_dimension_x)
        self.d_model_head = n_dimension_x // nhead
        self.extra_token_score_x = nn.Parameter(torch.randn(1, 2 * n_sampled_token, nhead), requires_grad=True)
        self.extra_token_q = nn.Parameter(torch.randn(1, nhead, 2 * n_sampled_token, n_dimension_qk // nhead), requires_grad=True)
        self.temperature = nn.Parameter(torch.ones(1, nhead, 1, 1) * temperature, requires_grad=True)
        self.grad_approx = nn.Parameter(torch.zeros(1, nhead, 1, n_dimension_qk // nhead), requires_grad=True)
        self.hard = hard

    def forward(self, x, q, mask):
        epsilon = 1e-20
        q = torch.cat([q, self.extra_token_q.expand(x.shape[0], -1, -1, -1)], dim=2)

        importance_scores = self.importance_score(x, mask)
        importance_scores = torch.cat([importance_scores, self.extra_token_score_x.expand(x.shape[0], -1, -1)], dim=1)
        importance_scores = get_gumbel(importance_scores)

        if self.training is True or self.hard is False:
            # Argsort and gather
            importance_scores, indices = torch.topk(importance_scores, self.n_sampled_token * 2, dim=1, largest=True, sorted=True)
            importance_scores = importance_scores.transpose(1, 2).unsqueeze(-1)
            indices = indices.transpose(1, 2).unsqueeze(-1)
            q = torch.take_along_dim(q, indices=indices, dim=2)
            q_top, q_bot = torch.split(q, split_size_or_sections=[self.n_sampled_token, self.n_sampled_token], dim=2)
            imp_top, imp_bot = torch.split(
                importance_scores, 
                split_size_or_sections=[self.n_sampled_token, importance_scores.shape[2] - self.n_sampled_token],
                dim=2)
            random_indices = torch.randn(x.shape[0], self.nhead, self.n_sampled_token, 1, device=x.device).argsort(dim=2)
            q_bot = torch.take_along_dim(q_bot, indices=random_indices, dim=2)
            imp_bot = torch.take_along_dim(imp_bot, indices=random_indices, dim=2)
            if self.hard is True:
                return HardChooseDenseGrad.apply(imp_top, imp_bot, q_top, q_bot, self.temperature, self.grad_approx)
            else:
                return SoftChooseDenseGrad.apply(imp_top, imp_bot, q_top, q_bot, self.temperature)
        
        else:
            _, indices_top = torch.topk(importance_scores, self.n_sampled_token, dim=1, largest=True, sorted=False)
            indices_top = indices_top.transpose(1, 2).unsqueeze(-1)
            q_top = torch.take_along_dim(q, indices=indices_top, dim=2)
            return q_top