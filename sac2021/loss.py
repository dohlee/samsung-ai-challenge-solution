import torch
import torch.nn as nn

# https://github.com/tuantle/regression-losses-pytorch/blob/master/regression_losses.py
class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))