import torch
from torch import nn


class PolynomialRegression(nn.Module):

    def __init__(self, order: int):
        super().__init__()
        self._order = order
        for i in range(order):
            self.__setattr__(f"p{i}", nn.Parameter(
                torch.randn(1, requires_grad=True, dtype=torch.float)))

    def forward(self, x):
        return sum(
            [self.__getattr__(f"p{i}") * (x ** i) for i in range(self._order)])
