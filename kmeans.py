import torch
from torch import nn as nn

"""
    Represents an abstract instance of a generalized kmeans approximation algorithm.
    Inherits from nn.Module from CUDA GPU support.
"""
class NaiveKmeans(nn.Module):
    """
        Constructor for an instance of a NaiveKmeans object.
        Args:
            - k : Integer; Number of clusters
            - x : torch.Tensor; subset of R^d
    """
    def __init__(self, k : int): # I'll think abt args later.
        super().__init__()
        self.k = k
        self.memory = {}
        #TODO add a tracker dict?

    # Mutates centers as a subset c of x, where |c| = k & each element in c is randomly sampled
    # according to uniform distribution.
    def _seeder(self, x : torch.Tensor):
        weights = torch.tensor([1./(x.size(0)) for point in x])
        indices = torch.multinomial(weights, self.k)
        centers = x[indices]
        return centers, indices

    """
       - [+] Cost function
       - [ ] forward function (Is clustering funciton)
    """
    def _cost(self, x : torch.Tensor, centers : torch.Tensor) -> (torch.Tensor, torch.Tensor): #tensor of costs, tensor of indices
        distances = torch.pow(torch.cdist(x, centers), 2) #Euclidean distance^2
        costs, indices = torch.min(distances, 1)
        return costs, indices

    # Clustering algorithm.
    def forward(self, x : torch.Tensor, *args) -> (torch.Tensor, torch.Tensor):
        centers, indices = args[0], [] if len(args) > 0 else self._seeder(x)
        curr_costs, indices = self._cost(x, centers)
        self.memory[curr_costs.sum().cpu().item()] = (centers, indices)
        curr_cost = float("inf")
        while True:
            centers = x[indices]
            curr_costs, indices = self._cost(x, centers)
            curr_cost = curr_costs.sum().cpu().item()
            if curr_cost ==  min(list(self.memory.keys())):
                print(f"Final : {curr_cost}, {centers}")
                return self.memory[curr_cost]
            else:
                self.memory[curr_cost] = (centers, indices)    
